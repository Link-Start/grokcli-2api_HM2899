package server

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hm2899/grokcli-2api/internal/accounts"
	"github.com/hm2899/grokcli-2api/internal/admin"
	adminauth "github.com/hm2899/grokcli-2api/internal/admin/auth"
	"github.com/hm2899/grokcli-2api/internal/auth"
	"github.com/hm2899/grokcli-2api/internal/buildinfo"
	"github.com/hm2899/grokcli-2api/internal/config"
	"github.com/hm2899/grokcli-2api/internal/integrations"
	"github.com/hm2899/grokcli-2api/internal/maintainer"
	"github.com/hm2899/grokcli-2api/internal/modelhealth"
	"github.com/hm2899/grokcli-2api/internal/models"
	"github.com/hm2899/grokcli-2api/internal/pool"
	"github.com/hm2899/grokcli-2api/internal/protocol/anthropic"
	"github.com/hm2899/grokcli-2api/internal/protocol/historycompact"
	"github.com/hm2899/grokcli-2api/internal/protocol/reasoning"
	"github.com/hm2899/grokcli-2api/internal/protocol/responses"
	"github.com/hm2899/grokcli-2api/internal/protocol/toolcall"
	"github.com/hm2899/grokcli-2api/internal/proxy"
	"github.com/hm2899/grokcli-2api/internal/quota"
	regclient "github.com/hm2899/grokcli-2api/internal/registration/client"
	"github.com/hm2899/grokcli-2api/internal/store/postgres"
	"github.com/hm2899/grokcli-2api/internal/store/redis"
	"github.com/hm2899/grokcli-2api/internal/upstream/grok"
)

type Options struct {
	Ready             func() bool
	Reason            func() string
	StaticDir         string
	PublicReadEnabled bool
	AdminReadEnabled  bool
	AdminWriteEnabled bool
	ChatEnabled       bool
	MessagesEnabled   bool
	ResponsesEnabled  bool
	APIKeys           *auth.APIKeyVerifier
	Models            *models.Catalog
	Store             *postgres.Connector
	// Candidates, when non-empty, is used by proxy routes instead of Store.ListPoolCandidates.
	// Intended for contract/e2e tests against a fake upstream.
	Candidates    []pool.Candidate
	AdminSessions admin.SessionVerifier
	PickObserver  proxy.PickObserver
	AffinityStore proxy.AffinityStore
	// Upstream is a shared Grok HTTP client (connection pool). Prefer this over
	// constructing a new client on every request.
	Upstream    *grok.Client
	Redis       *redis.Client
	Leader      *redis.Leader
	Maintainer  *maintainer.Service
	ModelHealth *modelhealth.Service
	Quota       *quota.Service
	Config      config.Config
	// RuntimeConfig is the live process config. When non-nil, request handlers
	// read streaming/tool policy from it so admin settings writes take effect
	// without restart. Config remains the startup snapshot.
	RuntimeConfig     *config.Config
	RegistrationURL   string
	RegistrationToken string
}

func (o Options) runtimeConfig() config.Config {
	if o.RuntimeConfig != nil {
		return *o.RuntimeConfig
	}
	return o.Config
}

func (o Options) applySettingsToRuntime(settings map[string]any) {
	if settings == nil {
		return
	}
	if o.RuntimeConfig != nil {
		o.RuntimeConfig.ApplyStoreSettings(settings)
	}
	// History compact knobs live in historycompact package (not Config struct).
	applyHistoryCompactSettings(settings)
}

func applyHistoryCompactSettings(settings map[string]any) {
	if settings == nil {
		return
	}
	opts := historycompact.ConfigureOpts{}
	if v, ok := settings["history_compact_enabled"].(bool); ok {
		opts.Enabled = &v
	}
	if n, ok := asSettingsIntLocal(settings["history_compact_auto_chars"]); ok {
		opts.AutoChars = &n
	}
	if n, ok := asSettingsIntLocal(settings["history_keep_tool_rounds"]); ok {
		opts.KeepToolRounds = &n
	}
	if n, ok := asSettingsIntLocal(settings["history_max_tool_result_chars"]); ok {
		opts.MaxToolResultChars = &n
	}
	if opts.Enabled != nil || opts.AutoChars != nil || opts.KeepToolRounds != nil || opts.MaxToolResultChars != nil {
		historycompact.ConfigureFull(opts)
	}
}

func asSettingsIntLocal(value any) (int, bool) {
	switch v := value.(type) {
	case float64:
		return int(v), true
	case float32:
		return int(v), true
	case int:
		return v, true
	case int64:
		return int(v), true
	case json.Number:
		i, err := v.Int64()
		return int(i), err == nil
	case string:
		i, err := strconv.Atoi(strings.TrimSpace(v))
		return i, err == nil
	default:
		return 0, false
	}
}

// NewMigrationMux exposes migration-safe process probes plus low-risk read-only
// shells. Compatibility proxy/admin API endpoints are added only after their
// Python wire contracts are frozen.
func NewMigrationMux(ready func() bool) http.Handler {
	return NewMux(Options{Ready: ready})
}

func NewMux(options Options) http.Handler {
	mux := http.NewServeMux()
	staticDir := options.StaticDir
	if strings.TrimSpace(staticDir) == "" {
		staticDir = "static"
	}

	mux.HandleFunc("GET /live", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":             true,
			"implementation": buildinfo.Implementation,
			"version":        buildinfo.Version,
		})
	})
	mux.HandleFunc("GET /ready", func(w http.ResponseWriter, _ *http.Request) {
		if !isReady(options) {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{
				"ok":             false,
				"implementation": buildinfo.Implementation,
				"version":        buildinfo.Version,
				"reason":         readyReason(options),
			})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":             true,
			"implementation": buildinfo.Implementation,
			"version":        buildinfo.Version,
		})
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		status := "ok"
		if !isReady(options) {
			status = "starting"
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"status":         status,
			"implementation": buildinfo.Implementation,
			"version":        buildinfo.Version,
			"ready":          status == "ok",
		})
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		ready := 0
		if isReady(options) {
			ready = 1
		}
		_, _ = w.Write([]byte("# HELP g2a_runtime_ready Go runtime readiness gate.\n"))
		_, _ = w.Write([]byte("# TYPE g2a_runtime_ready gauge\n"))
		_, _ = w.Write([]byte("g2a_runtime_ready{implementation=\"go\"} " + itoa(ready) + "\n"))
		_, _ = w.Write([]byte(streamMetricsPrometheus()))
	})
	// Exact root only. A bare "GET /" is a subtree pattern in Go 1.22+ and would
	// incorrectly serve index.html for every unmatched path (e.g. /unknown).
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		serveFile(w, r, filepath.Join(staticDir, "index.html"), false)
	})
	mux.HandleFunc("GET /favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		serveFile(w, r, filepath.Join(staticDir, "favicon.ico"), false)
	})
	mux.HandleFunc("GET /admin", func(w http.ResponseWriter, r *http.Request) {
		serveAdminPage(w, r, staticDir, "index")
	})
	mux.HandleFunc("GET /admin/{page}", func(w http.ResponseWriter, r *http.Request) {
		serveAdminPage(w, r, staticDir, r.PathValue("page"))
	})
	mux.HandleFunc("GET /static/{file...}", func(w http.ResponseWriter, r *http.Request) {
		serveStatic(w, r, staticDir, r.PathValue("file"))
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		serveModels(w, r, options)
	})
	mux.HandleFunc("GET /models", func(w http.ResponseWriter, r *http.Request) {
		serveModels(w, r, options)
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		serveChatCompletions(w, r, options)
	})
	mux.HandleFunc("POST /chat/completions", func(w http.ResponseWriter, r *http.Request) {
		serveChatCompletions(w, r, options)
	})
	mux.HandleFunc("POST /v1/messages", func(w http.ResponseWriter, r *http.Request) {
		serveMessages(w, r, options)
	})
	mux.HandleFunc("POST /messages", func(w http.ResponseWriter, r *http.Request) {
		serveMessages(w, r, options)
	})
	mux.HandleFunc("POST /v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		serveMessagesCountTokens(w, r, options)
	})
	mux.HandleFunc("POST /messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		serveMessagesCountTokens(w, r, options)
	})
	mux.HandleFunc("POST /v1/responses", func(w http.ResponseWriter, r *http.Request) {
		serveResponses(w, r, options)
	})
	mux.HandleFunc("POST /responses", func(w http.ResponseWriter, r *http.Request) {
		serveResponses(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/status", func(w http.ResponseWriter, r *http.Request) {
		serveAdminStatus(w, r, options, false)
	})
	mux.HandleFunc("GET /admin/api/dashboard", func(w http.ResponseWriter, r *http.Request) {
		serveAdminStatus(w, r, options, true)
	})
	mux.HandleFunc("GET /admin/api/models", func(w http.ResponseWriter, r *http.Request) {
		serveAdminModels(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/keys", func(w http.ResponseWriter, r *http.Request) {
		serveAdminKeys(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts", func(w http.ResponseWriter, r *http.Request) {
		serveAdminAccounts(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/settings", func(w http.ResponseWriter, r *http.Request) {
		serveAdminSettings(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/logs", func(w http.ResponseWriter, r *http.Request) {
		serveAdminLogs(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/logs/actions", func(w http.ResponseWriter, r *http.Request) {
		serveAdminLogActions(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/usage/summary", func(w http.ResponseWriter, r *http.Request) {
		serveUsageSummary(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/usage/series", func(w http.ResponseWriter, r *http.Request) {
		serveUsageSeries(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/usage/by-key", func(w http.ResponseWriter, r *http.Request) {
		serveUsageBreakdown(w, r, options, "key")
	})
	mux.HandleFunc("GET /admin/api/usage/by-account", func(w http.ResponseWriter, r *http.Request) {
		serveUsageBreakdown(w, r, options, "account")
	})
	mux.HandleFunc("GET /admin/api/usage/by-model", func(w http.ResponseWriter, r *http.Request) {
		serveUsageBreakdown(w, r, options, "model")
	})
	mux.HandleFunc("GET /admin/api/usage/events", func(w http.ResponseWriter, r *http.Request) {
		serveUsageEvents(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/setup", func(w http.ResponseWriter, r *http.Request) {
		serveAdminSetup(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/login", func(w http.ResponseWriter, r *http.Request) {
		serveAdminLogin(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/session", func(w http.ResponseWriter, r *http.Request) {
		serveAdminSession(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/logout", func(w http.ResponseWriter, r *http.Request) {
		serveAdminLogout(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/keys", func(w http.ResponseWriter, r *http.Request) {
		serveAdminCreateKey(w, r, options)
	})
	mux.HandleFunc("PATCH /admin/api/keys/{key_id}", func(w http.ResponseWriter, r *http.Request) {
		serveAdminUpdateKey(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/keys/{key_id}/regenerate", func(w http.ResponseWriter, r *http.Request) {
		serveAdminRegenerateKey(w, r, options)
	})
	mux.HandleFunc("DELETE /admin/api/keys/{key_id}", func(w http.ResponseWriter, r *http.Request) {
		serveAdminDeleteKey(w, r, options)
	})
	mux.HandleFunc("PATCH /admin/api/accounts/{account_id}/enabled", func(w http.ResponseWriter, r *http.Request) {
		serveAdminSetAccountEnabled(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/{account_id}/kick", func(w http.ResponseWriter, r *http.Request) {
		serveAdminKickAccount(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/{account_id}/cooldown/clear", func(w http.ResponseWriter, r *http.Request) {
		serveAdminClearCooldown(w, r, options)
	})
	mux.HandleFunc("PATCH /admin/api/accounts/{account_id}/status", func(w http.ResponseWriter, r *http.Request) {
		serveAdminSetAccountStatus(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/{account_id}/status", func(w http.ResponseWriter, r *http.Request) {
		serveAdminSetAccountStatus(w, r, options)
	})
	mux.HandleFunc("PUT /admin/api/settings", func(w http.ResponseWriter, r *http.Request) {
		serveAdminUpdateSettings(w, r, options)
	})
	mux.HandleFunc("PATCH /admin/api/settings", func(w http.ResponseWriter, r *http.Request) {
		serveAdminUpdateSettings(w, r, options)
	})
	mux.HandleFunc("PUT /admin/api/settings/runtime", func(w http.ResponseWriter, r *http.Request) {
		serveAdminUpdateSettings(w, r, options)
	})
	mux.HandleFunc("PATCH /admin/api/settings/runtime", func(w http.ResponseWriter, r *http.Request) {
		serveAdminUpdateSettings(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/register-email/availability", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationAvailability(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/register-email/config", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationConfigGet(w, r, options)
	})
	mux.HandleFunc("PUT /admin/api/accounts/register-email/config", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationConfigPut(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/register-email/test-proxy", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationProxyTest(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/register-email/test-proxy", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationProxyTest(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/register-email/sessions", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationSessions(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/register-email/sessions/{session_id}", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationSession(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/register-email/sessions/{session_id}/stop", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationStopSession(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/register-email/batches/{batch_id}", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationBatch(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/register-email/batches/{batch_id}/stop", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationStopBatch(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/register-email/batches/{batch_id}/resume", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationResumeBatch(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/register-email", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationStart(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/register-email/reclaim", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationReclaim(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/register-email/stop", func(w http.ResponseWriter, r *http.Request) {
		serveRegistrationStopAll(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/import-sso", func(w http.ResponseWriter, r *http.Request) {
		serveSSOImportStart(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/login", func(w http.ResponseWriter, r *http.Request) {
		serveDeviceLoginStart(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/login/sessions/{session_id}", func(w http.ResponseWriter, r *http.Request) {
		serveDeviceLoginSession(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/login/sessions", func(w http.ResponseWriter, r *http.Request) {
		serveDeviceLoginSessions(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/import-sso/jobs/{job_id}", func(w http.ResponseWriter, r *http.Request) {
		serveSSOImportJob(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/import", func(w http.ResponseWriter, r *http.Request) {
		serveAdminImportAccount(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/export", func(w http.ResponseWriter, r *http.Request) {
		serveAdminExportAccounts(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/export-batch", func(w http.ResponseWriter, r *http.Request) {
		serveAdminExportAccountsBatch(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/export/jobs/{job_id}", func(w http.ResponseWriter, r *http.Request) {
		serveExportJobStatus(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/export/jobs/{job_id}/download", func(w http.ResponseWriter, r *http.Request) {
		serveExportJobDownload(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/delete-batch", func(w http.ResponseWriter, r *http.Request) {
		serveAdminDeleteAccountsBatch(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/logout", func(w http.ResponseWriter, r *http.Request) {
		serveAdminClearAllAccounts(w, r, options)
	})
	mux.HandleFunc("DELETE /admin/api/accounts/{account_id}", func(w http.ResponseWriter, r *http.Request) {
		serveAdminDeleteAccount(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/{account_id}/probe", func(w http.ResponseWriter, r *http.Request) {
		serveAdminProbeAccount(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/probe-batch", func(w http.ResponseWriter, r *http.Request) {
		serveAdminProbeBatch(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/probe-all", func(w http.ResponseWriter, r *http.Request) {
		serveAdminProbeAll(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/model-health", func(w http.ResponseWriter, r *http.Request) {
		serveModelHealthStatus(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/maintainer", func(w http.ResponseWriter, r *http.Request) {
		serveMaintainerStatus(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/maintainer/run", func(w http.ResponseWriter, r *http.Request) {
		serveMaintainerRun(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/refresh", func(w http.ResponseWriter, r *http.Request) {
		serveAccountsRefresh(w, r, options)
	})
	mux.HandleFunc("PUT /admin/api/settings/token-maintain", func(w http.ResponseWriter, r *http.Request) {
		serveToggleTokenMaintain(w, r, options)
	})
	mux.HandleFunc("PUT /admin/api/settings/model-health", func(w http.ResponseWriter, r *http.Request) {
		serveToggleModelHealth(w, r, options)
	})
	mux.HandleFunc("PUT /admin/api/settings/account-mode", func(w http.ResponseWriter, r *http.Request) {
		serveSetAccountMode(w, r, options)
	})
	mux.HandleFunc("PUT /admin/api/settings/password", func(w http.ResponseWriter, r *http.Request) {
		serveChangeAdminPassword(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/model-blocks/prune", func(w http.ResponseWriter, r *http.Request) {
		servePruneModelBlocks(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/export-sso", func(w http.ResponseWriter, r *http.Request) {
		serveExportAccountsSSO(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/export-sso", func(w http.ResponseWriter, r *http.Request) {
		serveExportAccountsSSOSelected(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/import-file", func(w http.ResponseWriter, r *http.Request) {
		serveAdminImportFile(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/import-files", func(w http.ResponseWriter, r *http.Request) {
		serveAdminImportFiles(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/normalize", func(w http.ResponseWriter, r *http.Request) {
		serveAdminNormalizeAccounts(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/models/sync", func(w http.ResponseWriter, r *http.Request) {
		serveAdminModelsSync(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/quota", func(w http.ResponseWriter, r *http.Request) {
		serveAdminAccountsQuota(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/accounts/{account_id}/quota", func(w http.ResponseWriter, r *http.Request) {
		serveAdminAccountQuota(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/settings/cliproxyapi", func(w http.ResponseWriter, r *http.Request) {
		serveIntegrationSettingsGet(w, r, options, "cliproxyapi_config")
	})
	mux.HandleFunc("PUT /admin/api/settings/cliproxyapi", func(w http.ResponseWriter, r *http.Request) {
		serveIntegrationSettingsPut(w, r, options, "cliproxyapi_config")
	})
	mux.HandleFunc("POST /admin/api/settings/cliproxyapi/test", func(w http.ResponseWriter, r *http.Request) {
		serveCLIProxyTest(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/export-cliproxyapi-format", func(w http.ResponseWriter, r *http.Request) {
		serveExportCLIProxyFormat(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/push-cliproxyapi", func(w http.ResponseWriter, r *http.Request) {
		servePushCLIProxy(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/settings/sub2api", func(w http.ResponseWriter, r *http.Request) {
		serveIntegrationSettingsGet(w, r, options, "sub2api_config")
	})
	mux.HandleFunc("PUT /admin/api/settings/sub2api", func(w http.ResponseWriter, r *http.Request) {
		serveIntegrationSettingsPut(w, r, options, "sub2api_config")
	})
	mux.HandleFunc("POST /admin/api/accounts/export-sub2api-format", func(w http.ResponseWriter, r *http.Request) {
		serveExportSub2APIFormat(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/accounts/push-sub2api", func(w http.ResponseWriter, r *http.Request) {
		servePushSub2API(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/settings/sub2api/test", func(w http.ResponseWriter, r *http.Request) {
		serveSub2APITest(w, r, options)
	})
	mux.HandleFunc("GET /admin/api/settings/sub2api/groups", func(w http.ResponseWriter, r *http.Request) {
		serveSub2APIGroupsList(w, r, options)
	})
	mux.HandleFunc("POST /admin/api/settings/sub2api/groups", func(w http.ResponseWriter, r *http.Request) {
		serveSub2APIGroupsCreate(w, r, options)
	})
	return mux
}

func serveModels(w http.ResponseWriter, r *http.Request, options Options) {
	if !options.PublicReadEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "Go public read routes are not enabled"})
		return
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": readyReason(options)})
		return
	}
	if options.APIKeys != nil {
		if _, err := options.APIKeys.Require(r.Context(), r); err != nil {
			status := http.StatusInternalServerError
			message := err.Error()
			if errors.Is(err, auth.ErrInvalidAPIKey) {
				status = http.StatusUnauthorized
				message = "Invalid or missing API key"
			}
			writeJSON(w, status, map[string]any{"detail": message})
			return
		}
	}
	catalog := options.Models
	if catalog == nil {
		catalog = models.NewCatalog(config.Config{DefaultModel: "grok-4.5"}, nil)
	}
	writeJSON(w, http.StatusOK, catalog.OpenAIList(r.Context()))
}

func serveChatCompletions(w http.ResponseWriter, r *http.Request, options Options) {
	if !options.ChatEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go chat route is not enabled"})
		return
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return
	}
	var apiKey *auth.APIKeyRecord
	if options.APIKeys != nil {
		verified, err := options.APIKeys.Require(r.Context(), r)
		if err != nil {
			status := http.StatusInternalServerError
			message := err.Error()
			if errors.Is(err, auth.ErrInvalidAPIKey) {
				status = http.StatusUnauthorized
				message = "Invalid or missing API key"
			}
			writeJSON(w, status, map[string]any{"detail": message})
			return
		}
		apiKey = verified
	}
	if options.Store == nil && len(options.Candidates) == 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	chatReq, err := proxy.DecodeChatRequest(r.Body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	chatReq.UserAgent = r.UserAgent()

	// Codex shell schema uses "cmd"; remember preferred keys from the client tools
	// so outbound tool_calls project command→cmd (or honor pure command-only schemas).
	keys := toolcall.ShellArgKeyMap(chatReq.Raw["tools"])
	keys = ensureCodexShellCmdKeys(chatReq.Raw["tools"], keys)
	if keys == nil {
		keys = map[string]string{}
	}
	// Proxies often strip User-Agent — also detect Codex via tools schema (exec_command/cmd).
	if len(keys) == 0 && historycompact.LooksLikeCodexRequest(r.UserAgent(), chatReq.Raw["tools"], chatReq.Raw) {
		for _, name := range []string{"exec_command", "run_command", "shell", "bash", "local_shell", "shell_command", "Shell", "Bash", "localShell"} {
			keys[name] = "cmd"
			keys[strings.ToLower(name)] = "cmd"
			if nk := toolcall.NameKey(name); nk != "" {
				keys[nk] = "cmd"
			}
		}
	}
	if len(keys) > 0 {
		r = r.WithContext(withShellArgKeys(r.Context(), keys))
	}
	// Client-registered tools for Update/StrReplace → Edit remap on chat path.
	if allowed := allowedAnthropicToolNames(chatReq.Raw); len(allowed) > 0 {
		r = r.WithContext(withAllowedTools(r.Context(), allowed))
	}
	candidates, err := listCandidatesForRequest(r.Context(), options, chatReq, r.Header)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	service := newChatService(options)
	started := time.Now()
	if chatReq.Stream {
		opened, err := service.OpenStreamWithResult(r.Context(), chatReq, candidates, resolvePickMode(options))
		if err != nil {
			// Intermediate + final open failures are already classified into the
			// cooldown pool via ChatService.FailureReporter (text-based 额度用完).
			recordChatUsage(r, options, apiKey, opened.AccountID, chatReq.Model, chatReq.Stream, false, http.StatusBadGateway, started, nil, err, 0, chatReq.Raw)
			writeProxyError(w, err)
			return
		}
		defer opened.Body.Close()
		defer releaseServerPick(options, opened.AccountID)
		req := r
		if options.runtimeConfig().SSEKeepalive > 0 {
			req = r.WithContext(withAnthropicKeepalive(r.Context(), options.runtimeConfig().SSEKeepalive))
		}
		setProtocolObservationHeaders(w, protocolObservation{
			Protocol: "openai_chat", AccountID: opened.AccountID, PreferAccount: opened.PreferAccount,
			Failover: opened.Failover, Fingerprint: opened.Fingerprint, Accounts: opened.Accounts, Prep: opened.Prep,
		})
		stats, err := streamChatCompletions(w, req, opened.Body, optionsFromRequest(req).Keepalive)
		ok := err == nil || errors.Is(err, r.Context().Err())
		status := http.StatusOK
		if !ok {
			status = http.StatusBadGateway
		}
		recordChatUsage(r, options, apiKey, opened.AccountID, opened.Model, chatReq.Stream, ok, status, started, stats.Usage, err, stats.FirstTokenMS, chatReq.Raw)
		reportChatPool(r, options, opened.AccountID, ok, err, status, opened.Model)
		return
	}
	result, err := service.CompleteWithResult(r.Context(), chatReq, candidates, resolvePickMode(options))
	if result.AccountID != "" {
		defer releaseServerPick(options, result.AccountID)
	}
	if err != nil {
		// Failures already reported per-account via FailureReporter.
		recordChatUsage(r, options, apiKey, result.AccountID, chatReq.Model, chatReq.Stream, false, http.StatusBadGateway, started, nil, err, 0, chatReq.Raw)
		writeProxyError(w, err)
		return
	}
	// Project shell args for non-stream chat completions (Codex cmd / OpenAI command).
	if result.Payload != nil && len(keys) > 0 {
		projectChatPayloadShellArgs(result.Payload, keys)
	} else if result.Payload != nil {
		// Default shell-family tools to cmd even without an explicit map.
		projectChatPayloadShellArgs(result.Payload, nil)
	}
	recordChatUsage(r, options, apiKey, result.AccountID, result.Model, chatReq.Stream, true, http.StatusOK, started, result.Usage, nil, 0, chatReq.Raw)
	reportChatPool(r, options, result.AccountID, true, nil, http.StatusOK, result.Model)
	setProtocolObservationHeaders(w, protocolObservation{
		Protocol: "openai_chat", AccountID: result.AccountID, PreferAccount: result.PreferAccount,
		Failover: result.Failover, Fingerprint: result.Fingerprint, Accounts: result.Accounts, Prep: result.Prep,
	})
	writeJSON(w, http.StatusOK, result.Payload)
}

// projectChatPayloadShellArgs rewrites shell tool_calls / function_call arguments
// in a non-stream chat.completion payload to the client preferred key (cmd/command).
func projectChatPayloadShellArgs(payload map[string]any, keys map[string]string) {
	if payload == nil {
		return
	}
	choices, _ := payload["choices"].([]any)
	if choices == nil {
		// collector uses []map[string]any
		if typed, ok := payload["choices"].([]map[string]any); ok {
			for _, ch := range typed {
				projectChatChoiceShellArgs(ch, keys)
			}
		}
		return
	}
	for _, item := range choices {
		ch, _ := item.(map[string]any)
		projectChatChoiceShellArgs(ch, keys)
	}
}

func projectChatChoiceShellArgs(choice map[string]any, keys map[string]string) {
	if choice == nil {
		return
	}
	msg, _ := choice["message"].(map[string]any)
	if msg == nil {
		return
	}
	if calls, ok := msg["tool_calls"].([]any); ok {
		for _, raw := range calls {
			call, _ := raw.(map[string]any)
			projectOneChatToolCall(call, keys)
		}
	} else if typed, ok := msg["tool_calls"].([]map[string]any); ok {
		for _, call := range typed {
			projectOneChatToolCall(call, keys)
		}
	}
	if fn, ok := msg["function_call"].(map[string]any); ok && fn != nil {
		name := strings.TrimSpace(fmt.Sprint(fn["name"]))
		args := strings.TrimSpace(fmt.Sprint(fn["arguments"]))
		if name != "" && args != "" && toolcall.IsShellTool(name) {
			pref := "cmd"
			if keys != nil {
				if v := strings.TrimSpace(keys[name]); v != "" {
					pref = v
				} else if v := strings.TrimSpace(keys[strings.ToLower(name)]); v != "" {
					pref = v
				} else if nk := toolcall.NameKey(name); nk != "" {
					if v := strings.TrimSpace(keys[nk]); v != "" {
						pref = v
					}
				}
			}
			fn["arguments"] = toolcall.ProjectShellArgsForClient(args, name, pref)
		}
	}
}

func projectOneChatToolCall(call map[string]any, keys map[string]string) {
	if call == nil {
		return
	}
	fn, _ := call["function"].(map[string]any)
	if fn == nil {
		return
	}
	name := strings.TrimSpace(fmt.Sprint(fn["name"]))
	args := strings.TrimSpace(fmt.Sprint(fn["arguments"]))
	if name == "" || args == "" || !toolcall.IsShellTool(name) {
		return
	}
	pref := "cmd"
	if keys != nil {
		if v := strings.TrimSpace(keys[name]); v != "" {
			pref = v
		} else if v := strings.TrimSpace(keys[strings.ToLower(name)]); v != "" {
			pref = v
		} else if nk := toolcall.NameKey(name); nk != "" {
			if v := strings.TrimSpace(keys[nk]); v != "" {
				pref = v
			}
		}
	}
	fn["arguments"] = toolcall.ProjectShellArgsForClient(args, name, pref)
}

func streamChatCompletions(w http.ResponseWriter, r *http.Request, body io.Reader, keepalive time.Duration) (proxy.StreamStats, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": "streaming is not supported by this response writer"})
		return proxy.StreamStats{}, errors.New("streaming is not supported by this response writer")
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	var stats proxy.StreamStats
	started := time.Now()
	// Assemble tool_calls across chunks so we can emit normalized args once complete
	// (alias rewrite + drop incomplete tools) without breaking partial JSON streams.
	assembler := proxy.NewChatToolStreamAssembler()
	// Project shell args to client schema (Codex: cmd) using keys from serveChatCompletions.
	assembler.SetShellArgKeys(shellArgKeysFrom(r.Context()))
	// Remap Grok Update/StrReplace → client Edit when tools are known.
	assembler.SetAllowedTools(allowedToolNamesFrom(r.Context()))
	// Soft client write/ctx blips must NOT abort ReadSSE: aborting drains the
	// remaining upstream tool-arg frames, force-finishes with half args, and
	// Claude Code surfaces that as intermittent "Tool use interrupted".
	sw := newSSEWriter(w, flusher, r.Context())
	// Coalesce pure text/reasoning passthrough frames to cut Flush storms.
	pending := make([]byte, 0, 1024)
	firstPayloadFlushed := false
	flushPending := func(force bool) error {
		if len(pending) == 0 {
			return nil
		}
		payload := pending
		pending = pending[:0]
		streamCoalesceFlush.Add(1)
		if err := sw.WriteBytes(payload, force || sw.SoftGone()); err != nil {
			return err
		}
		if sw.LastOK() {
			firstPayloadFlushed = true
		}
		return nil
	}
	writeJSONFrame := func(payload map[string]any, force bool) error {
		if err := flushPending(true); err != nil {
			return err
		}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		frame := make([]byte, 0, len(encoded)+8)
		frame = append(frame, "data: "...)
		frame = append(frame, encoded...)
		frame = append(frame, '\n', '\n')
		if err := sw.WriteBytes(frame, force); err != nil {
			return err
		}
		if sw.LastOK() {
			assembler.AckPayload(string(encoded))
		} else if assembler.HasUnacked() {
			assembler.RequeueUnacked()
		}
		return nil
	}
	terminalFlushed := false
	// Real model payload only (text/reasoning/tools). Role-only / empty / usage-only
	// SSE must NOT count — that was an intermittent "soft-success empty" path when
	// upstream dropped after hollow frames (FirstTokenMS>0 with no client content).
	sawRealPayload := false
	sawFinishReason := false
	streamID := ""
	streamModel := ""
	flushAssemblerTerminal := func() error {
		_ = flushPending(true)
		if terminalFlushed && !assembler.NeedsFinishRetry() {
			return nil
		}
		// End-of-stream / soft disconnect: force-finish incomplete tools and always
		// emit a finish_reason frame so clients do not hang as "Tool use interrupted"
		// or incomplete mid-response (text-only turns included).
		for attempt := 0; attempt < 5; attempt++ {
			terminalFlushed = true
			for _, frame := range assembler.Finish() {
				if err := writeJSONFrame(frame, true); err != nil {
					return err
				}
			}
			if frame := assembler.FinishReasonFrame(); frame != nil {
				if err := writeJSONFrame(frame, true); err != nil {
					return err
				}
				sawFinishReason = true
			} else if !sawFinishReason {
				// Text-only soft terminal (or tools all dropped): still emit finish_reason
				// so OpenAI clients / Claude Code leave "running" cleanly.
				id := streamID
				if id == "" {
					id = "chatcmpl-go-stream"
				}
				model := streamModel
				if model == "" {
					model = "grok-4.5"
				}
				term := map[string]any{
					"id":      id,
					"object":  "chat.completion.chunk",
					"created": time.Now().Unix(),
					"model":   model,
					"choices": []any{
						map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"},
					},
				}
				if err := writeJSONFrame(term, true); err != nil {
					return err
				}
				sawFinishReason = true
			}
			if !assembler.NeedsFinishRetry() {
				break
			}
		}
		return sw.WriteBytes([]byte("data: [DONE]\n\n"), true)
	}
	err := grok.ReadSSEWithIdle(body, keepalive, func(event grok.Event) error {
		if event.Done {
			return flushAssemblerTerminal()
		}
		delta, err := proxy.ParseChatDelta(event.Data)
		if err == nil && delta.Usage != nil {
			stats.Usage = delta.Usage
		}
		if err == nil {
			if delta.ID != "" {
				streamID = delta.ID
			}
			if delta.Model != "" {
				streamModel = delta.Model
			}
			hasContent := strings.TrimSpace(delta.Content) != "" || strings.TrimSpace(delta.Reasoning) != ""
			hasTools := len(delta.ToolCalls) > 0 || delta.FunctionCall != nil
			if hasContent || hasTools {
				sawRealPayload = true
				// TTFT = first real model payload, not hollow role/usage SSE.
				if stats.FirstTokenMS == 0 {
					stats.FirstTokenMS = int(time.Since(started).Milliseconds())
					if stats.FirstTokenMS <= 0 {
						stats.FirstTokenMS = 1
					}
				}
			}
			if delta.FinishReason != nil {
				sawFinishReason = true
			}
		}
		// When tool deltas present (or assembler is already buffering tools), rewrite
		// through assembler; pure text/finish frames passthrough with coalesce.
		if err == nil && (len(delta.ToolCalls) > 0 || delta.FunctionCall != nil || assembler.Holding()) {
			frames, passthrough := assembler.Feed(event.Data, delta)
			if !passthrough {
				// Once tools are in flight, force writes so a soft disconnect still
				// delivers complete tool_calls + finish_reason rather than half frames.
				writeForce := assembler.Holding() || assembler.EmittedAny() || sw.SoftGone()
				for _, frame := range frames {
					if err := writeJSONFrame(frame, writeForce); err != nil {
						return err
					}
				}
				return nil
			}
		}
		// Coalesce pure text passthrough: first token immediate (TTFT), then batch
		// until textCoalesceMax / tool / idle / finish.
		frame := make([]byte, 0, len(event.Data)+8)
		frame = append(frame, "data: "...)
		frame = append(frame, event.Data...)
		frame = append(frame, '\n', '\n')
		// Force first real payload for TTFT; coalesce subsequent pure text.
		canCoalesce := firstPayloadFlushed && !assembler.Holding() && !assembler.EmittedAny() && !sw.SoftGone()
		if canCoalesce {
			pending = append(pending, frame...)
			if len(pending) >= textCoalesceMax {
				return flushPending(false)
			}
			return nil
		}
		if err := flushPending(true); err != nil {
			return err
		}
		if err := sw.WriteBytes(frame, true); err != nil {
			return err
		}
		if sw.LastOK() && sawRealPayload {
			firstPayloadFlushed = true
		}
		return nil
	}, func() error {
		_ = flushPending(true)
		select {
		case <-r.Context().Done():
			sw.MarkSoftGone()
			return sw.Keepalive(": keepalive\n\n", DefaultKeepaliveInterval, true)
		default:
		}
		return sw.Keepalive(": keepalive\n\n", DefaultKeepaliveInterval, false)
	})
	// Soft disconnect / write abort / mid-stream upstream drop after tools or
	// content started: force-finish so clients do not hang, and avoid a second
	// error JSON that Claude Code reports as "Server error mid-response".
	_ = flushPending(true)
	hasStreamPayload := assembler.Holding() || assembler.EmittedAny() || sawRealPayload
	clientSoft := sw.SoftGone() || errors.Is(err, r.Context().Err()) || isSoftClientWriteError(err)
	upstreamMid := err != nil && !clientSoft
	if hasStreamPayload {
		if err == nil || clientSoft || upstreamMid {
			_ = flushAssemblerTerminal()
			if clientSoft || upstreamMid {
				return stats, nil
			}
		}
	}
	if err != nil && !errors.Is(err, r.Context().Err()) && !hasStreamPayload {
		msg, errType := openAIErrorFromCause(err)
		encoded, _ := json.Marshal(map[string]any{"error": map[string]any{"message": msg, "type": errType}})
		_ = sw.WriteBytes(append(append([]byte("data: "), encoded...), '\n', '\n'), true)
		_ = sw.WriteBytes([]byte("data: [DONE]\n\n"), true)
	}
	return stats, err
}

func releaseServerPick(options Options, accountID string) {
	if options.PickObserver == nil || accountID == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	options.PickObserver.ReleasePick(ctx, accountID)
}

func recordChatUsage(r *http.Request, options Options, apiKey *auth.APIKeyRecord, accountID, model string, stream bool, ok bool, status int, started time.Time, usage any, cause error, ttftMS int, requestBody map[string]any) {
	prompt, completion, total, cacheRead, cacheCreate, reasoning := postgres.UsageFromOpenAI(usage)
	streamValue := stream
	var apiKeyID string
	if apiKey != nil {
		apiKeyID = apiKey.ID
	}
	var errText string
	if cause != nil {
		errText = cause.Error()
	}
	latency := int(time.Since(started).Milliseconds())
	var ttftPtr *int
	if ttftMS > 0 {
		v := ttftMS
		ttftPtr = &v
	}
	// Fire-and-forget with longer timeout - usage recording should not block response
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if options.Store != nil {
			if _, _, err := options.Store.RecordUsage(ctx, postgres.UsageRecord{
				RequestID:           requestID(r),
				Implementation:      "go",
				APIKeyID:            apiKeyID,
				AccountID:           accountID,
				Model:               model,
				Protocol:            "openai_chat",
				Path:                r.URL.Path,
				Stream:              &streamValue,
				OK:                  ok,
				PromptTokens:        prompt,
				CompletionTokens:    completion,
				TotalTokens:         total,
				CacheReadTokens:     cacheRead,
				CacheCreationTokens: cacheCreate,
				ReasoningTokens:     reasoning,
				ClientIP:            clientIP(r),
				UserAgent:           r.UserAgent(),
				StatusCode:          &status,
				LatencyMS:           &latency,
				TTFTMS:              ttftPtr,
				Error:               errText,
				Detail:              usageDetail("go_chat", requestBody, ttftMS, latency),
			}); err != nil {
				slog.Warn("record usage failed", "error", err, "account_id", accountID, "model", model, "ok", ok)
			}
		}
		recordRedisUsage(options, apiKeyID, accountID, model, prompt, completion, total, cacheRead, ok)
	}()
}

func reportChatPool(r *http.Request, options Options, accountID string, ok bool, cause error, status int, requestedModel ...string) {
	if strings.TrimSpace(accountID) == "" {
		return
	}
	modelHint := ""
	if len(requestedModel) > 0 {
		modelHint = strings.TrimSpace(requestedModel[0])
	}
	// Fire-and-forget pool reporting - should not block response
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if ok {
			if options.Store != nil {
				// preserve still-active free-usage windows, but heal stale markers.
				_ = options.Store.ReportPoolSuccess(ctx, accountID, true)
			}
			touchRedisPool(options, accountID, true, "", nil, status)
			return
		}
		var errText string
		if cause != nil {
			errText = cause.Error()
		}
		// Prefer upstream status/body when present (failover wraps).
		var upstream *grok.UpstreamError
		if errors.As(cause, &upstream) && upstream != nil {
			if upstream.Status > 0 {
				status = upstream.Status
			}
			if strings.TrimSpace(upstream.Body) != "" {
				errText = upstream.Body
			}
		}
		// Text-first classifier: free-usage / 额度用完 / rate-limit wording decides
		// whether the account enters the cooldown pool (and optionally soft-blocks a model).
		decision := pool.ClassifyUpstreamFailure(status, errText, modelHint)
		var cooldown *time.Time
		if decision.ShouldCooldown {
			cooldown = decision.Until
		}
		// Prefer body-extracted model, fall back to requested model for soft-block.
		coolModel := strings.TrimSpace(decision.Model)
		if coolModel == "" {
			coolModel = modelHint
		}
		failure := postgres.PoolFailure{
			AccountID:      accountID,
			Error:          errText,
			StatusCode:     &status,
			CooldownUntil:  cooldown,
			CooldownReason: firstNonEmptyStr(decision.Reason, errText),
			CooldownCode:   decision.Code,
			CooldownModel:  coolModel,
			Detail: map[string]any{
				"source":           "go_chat",
				"failure_class":    string(decision.Class),
				"should_cooldown":  decision.ShouldCooldown,
				"requested_model":  modelHint,
				"classified_model": decision.Model,
			},
		}
		if decision.TokensActual != nil {
			failure.CooldownTokensActual = decision.TokensActual
		}
		if decision.TokensLimit != nil {
			failure.CooldownTokensLimit = decision.TokensLimit
		}
		// Soft-block model only when classifier sets BlockModel (not free-usage cool).
		if decision.BlockModel && coolModel != "" && cooldown != nil {
			failure.BlockedModel = coolModel
			failure.BlockedUntil = cooldown
		}
		if options.Store != nil {
			// Only record durable failure/cooldown when classifier says so.
			// Still bump fail stats for non-cooldown errors without setting until.
			_ = options.Store.ReportPoolFailure(ctx, failure)
		}
		touchRedisPool(options, accountID, false, errText, cooldown, status)
	}()
}

// chatFailureCooldown is kept as a thin wrapper around pool.ClassifyUpstreamFailure
// for unit tests and call sites that still use the old signature.
func chatFailureCooldown(status int, errText string, requestedModel ...string) (until *time.Time, code, model string, tokensActual, tokensLimit *int64) {
	d := pool.ClassifyUpstreamFailure(status, errText, requestedModel...)
	if !d.ShouldCooldown {
		return nil, d.Code, d.Model, d.TokensActual, d.TokensLimit
	}
	return d.Until, d.Code, d.Model, d.TokensActual, d.TokensLimit
}

// chatPoolFailureReporter implements proxy.AccountFailureReporter so intermediate
// failover losers (quota-exhausted accounts) still enter the cooldown pool even
// when a later account eventually serves the request.
type chatPoolFailureReporter struct {
	options Options
}

func (r chatPoolFailureReporter) ReportAccountFailure(accountID, model string, err error) {
	if err == nil || strings.TrimSpace(accountID) == "" {
		return
	}
	status := http.StatusBadGateway
	reportChatPool(nil, r.options, accountID, false, err, status, model)
}

func newChatService(options Options) proxy.ChatService {
	return proxy.ChatService{
		Catalog:               modelCatalog(options),
		Client:                upstreamClient(options),
		PickObserver:          options.PickObserver,
		AffinityStore:         options.AffinityStore,
		FailureReporter:       chatPoolFailureReporter{options: options},
		StickyFirstOnly:       options.runtimeConfig().StickyFirstOnly,
		FirstByteProbeWorkers: options.runtimeConfig().FirstByteProbeWorkers,
	}
}

func extractGrokModelFromError(errText string) string {
	return pool.ExtractModelName(errText)
}

func parseTokenPair(errText string) (actual, limit int64, ok bool) {
	return pool.ParseTokenPair(errText)
}

func requestID(r *http.Request) string {
	for _, name := range []string{"X-Request-ID", "X-Correlation-ID", "X-Client-Request-ID"} {
		if value := strings.TrimSpace(r.Header.Get(name)); value != "" {
			return value
		}
	}
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "go-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	return "go-" + hex.EncodeToString(buf)
}

func clientIP(r *http.Request) string {
	if forwarded := strings.TrimSpace(r.Header.Get("X-Forwarded-For")); forwarded != "" {
		return strings.TrimSpace(strings.Split(forwarded, ",")[0])
	}
	if realIP := strings.TrimSpace(r.Header.Get("X-Real-IP")); realIP != "" {
		return realIP
	}
	return r.RemoteAddr
}

func writeProxyError(w http.ResponseWriter, err error) {
	// Keep OpenAI chat /v1/chat/completions error shape for this route.
	writeOpenAIProxyError(w, err)
}

// writeOpenAIProxyError maps upstream/proxy failures onto OpenAI error JSON
// with useful status codes and unwrapped human/quota messages.
func writeOpenAIProxyError(w http.ResponseWriter, err error) {
	status := http.StatusBadGateway
	message, errorType := openAIErrorFromCause(err)
	if errors.Is(err, pool.ErrNoEligibleAccounts) {
		status = http.StatusServiceUnavailable
	}
	var upstream *grok.UpstreamError
	if errors.As(err, &upstream) && upstream != nil {
		status = mapUpstreamStatusToOpenAI(upstream.Status)
	} else if unwrappedStatus, _, ok := unwrapAnthropicUpstreamMessage(errString(err)); ok && unwrappedStatus > 0 {
		// Reuse the same upstream-status peel helper (shared wrapper format).
		status = mapUpstreamStatusToOpenAI(unwrappedStatus)
	}
	writeOpenAIError(w, status, message, errorType)
}

func writeOpenAIError(w http.ResponseWriter, status int, message, errorType string) {
	if status <= 0 {
		status = http.StatusBadGateway
	}
	// Unwrap "upstream status N: {...}" and nested JSON error bodies.
	if unwrappedStatus, unwrappedBody, ok := unwrapAnthropicUpstreamMessage(message); ok {
		if unwrappedStatus > 0 {
			status = mapUpstreamStatusToOpenAI(unwrappedStatus)
		}
		if strings.TrimSpace(unwrappedBody) != "" {
			message = unwrappedBody
		}
		if errorType == "" || errorType == "server_error" || errorType == "api_error" {
			errorType = openAIErrorTypeForStatus(status)
		}
	}
	if errorType == "" {
		errorType = openAIErrorTypeForStatus(status)
	}
	if strings.TrimSpace(message) == "" {
		message = "request failed"
	}
	// OpenAI wire shape: { "error": { "message":"...", "type":"..." } }
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": message, "type": errorType}})
}

func openAIErrorTypeForStatus(status int) string {
	switch status {
	case http.StatusUnauthorized:
		return "invalid_request_error"
	case http.StatusForbidden:
		return "invalid_request_error"
	case http.StatusNotFound:
		return "invalid_request_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge:
		return "invalid_request_error"
	default:
		return "server_error"
	}
}

func mapUpstreamStatusToOpenAI(status int) int {
	switch {
	case status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable:
		return status
	case status == http.StatusUnauthorized || status == http.StatusForbidden || status == http.StatusNotFound:
		return status
	case status == http.StatusBadRequest || status == http.StatusRequestEntityTooLarge:
		return status
	case status >= 500:
		return http.StatusBadGateway
	case status >= 400:
		return status
	default:
		return http.StatusBadGateway
	}
}

func openAIErrorFromCause(err error) (message, errorType string) {
	if err == nil {
		return "request failed", "server_error"
	}
	message = strings.TrimSpace(err.Error())
	errorType = "server_error"
	var upstream *grok.UpstreamError
	if errors.As(err, &upstream) && upstream != nil {
		if strings.TrimSpace(upstream.Body) != "" {
			message = preferAnthropicErrorBody(upstream.Body)
		}
		status := mapUpstreamStatusToOpenAI(upstream.Status)
		errorType = openAIErrorTypeForStatus(status)
		return firstNonEmptyStr(message, err.Error()), errorType
	}
	if status, body, ok := unwrapAnthropicUpstreamMessage(message); ok {
		if strings.TrimSpace(body) != "" {
			message = body
		}
		if status > 0 {
			errorType = openAIErrorTypeForStatus(mapUpstreamStatusToOpenAI(status))
		}
	}
	low := strings.ToLower(message)
	if strings.Contains(low, "empty model output") || strings.Contains(low, "no content/tool_calls") {
		errorType = "server_error"
	}
	if errors.Is(err, pool.ErrNoEligibleAccounts) {
		errorType = "server_error"
		if message == "" || strings.Contains(low, "no eligible") {
			message = "No eligible accounts available. All accounts may be in cooldown or disabled."
		}
	}
	return firstNonEmptyStr(message, "request failed"), errorType
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func serveMessages(w http.ResponseWriter, r *http.Request, options Options) {
	apiKey, ok := messageRouteAllowed(w, r, options)
	if !ok {
		return
	}
	// Accepted for client compatibility (Claude SDKs send it); not enforced.
	_ = r.Header.Get("anthropic-version")
	// anthropic-beta: prompt-caching-* is accepted (sticky routing + usage fields).
	_ = r.Header.Get("anthropic-beta")
	var raw map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&raw); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	messages, _ := raw["messages"].([]any)
	if len(messages) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "messages: Field required", "invalid_request_error")
		return
	}
	if !positiveNumber(raw["max_tokens"]) {
		writeAnthropicError(w, http.StatusBadRequest, "max_tokens: Input should be greater than or equal to 1", "invalid_request_error")
		return
	}
	stream, _ := raw["stream"].(bool)
	model := modelCatalog(options).Resolve(stringValue(raw["model"]))
	body, err := anthropic.BuildOpenAIChatBody(raw, model)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	// Echo sticky cache key so clients / relays can reuse it next turn.
	if pck := strings.TrimSpace(stringValue(body["prompt_cache_key"])); pck != "" {
		w.Header().Set("X-Grok2API-Prompt-Cache-Key", pck)
	}
	allowedTools := allowedAnthropicToolNames(body)
	chatReq := proxy.ChatRequest{Model: model, Stream: false, Raw: body, UserAgent: r.UserAgent()}
	candidates, err := listCandidatesForRequest(r.Context(), options, chatReq, r.Header)
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, err.Error(), "api_error")
		return
	}
	service := newChatService(options)
	started := time.Now()
	messageID := newAnthropicMessageID()
	// Match Python resolve_outbound_max_tools for anthropic (Claude/sub2api-safe).
	policy := historycompact.ResolveOutboundToolPolicy(
		"anthropic",
		r.UserAgent(),
		options.runtimeConfig().OutboundMaxTools,
		options.runtimeConfig().OutboundMaxToolsOpenAI,
		options.runtimeConfig().OutboundMaxToolsResponsesNative,
		options.runtimeConfig().OutboundToolGap,
		options.runtimeConfig().OutboundToolGapNative,
	)
	maxTools := policy.MaxTools
	if maxTools < 0 {
		maxTools = 0
	}
	if stream {
		chatReq.Stream = true
		opened, err := service.OpenStreamWithResult(r.Context(), chatReq, candidates, resolvePickMode(options))
		if err != nil {
			// Per-account free-usage / 额度用完 already kicked via FailureReporter.
			recordAnthropicUsage(r, options, apiKey, opened.AccountID, model, true, false, http.StatusBadGateway, started, nil, err, 0, raw)
			writeAnthropicProxyError(w, err)
			return
		}
		defer opened.Body.Close()
		defer releaseServerPick(options, opened.AccountID)
		req := r
		if options.runtimeConfig().SSEKeepalive > 0 {
			req = r.WithContext(withAnthropicKeepalive(r.Context(), options.runtimeConfig().SSEKeepalive))
		}
		if policy.ToolGap > 0 {
			req = req.WithContext(withOutboundToolGap(req.Context(), policy.ToolGap))
		}
		setAnthropicObservationHeaders(w, protocolObservation{Protocol: "anthropic",
			AccountID: opened.AccountID, PreferAccount: opened.PreferAccount, Failover: opened.Failover,
			Fingerprint: opened.Fingerprint, Accounts: opened.Accounts, Prep: opened.Prep, Stream: true,
		})
		if pck := strings.TrimSpace(stringValue(body["prompt_cache_key"])); pck != "" {
			w.Header().Set("X-Grok2API-Prompt-Cache-Key", pck)
		}
		usage, firstTokenMS, err := streamAnthropicMessages(w, req, opened.Body, messageID, opened.Model, len(allowedTools) > 0, allowedTools, maxTools)
		// Empty upstream / tool-vanished returns an error; soft disconnect without
		// payload also returns empty error (see streamAnthropicMessagesWithOptions).
		// Only pure client-cancel AFTER payload is treated as ok.
		ok := err == nil
		if !ok && errors.Is(err, r.Context().Err()) {
			ok = true
		}
		status := http.StatusOK
		if !ok {
			status = http.StatusBadGateway
		}
		// Hollow / half-open tool streams return an empty error from the stream
		// function (ClientDeliveryOK / HasClientPayload). Do NOT flip ok=false
		// solely for zero usage tokens — upstream often omits usage on short tool
		// turns; that would false-fail real deliveries (and mask the real fix).
		recordAnthropicUsage(r, options, apiKey, opened.AccountID, opened.Model, true, ok, status, started, usage, err, firstTokenMS, raw)
		reportChatPool(r, options, opened.AccountID, ok, err, status, opened.Model)
		return
	}
	result, err := service.CompleteWithResult(r.Context(), chatReq, candidates, resolvePickMode(options))
	if result.AccountID != "" {
		defer releaseServerPick(options, result.AccountID)
	}
	if err != nil {
		// Per-account free-usage / 额度用完 already kicked via FailureReporter.
		recordAnthropicUsage(r, options, apiKey, result.AccountID, model, false, false, http.StatusBadGateway, started, nil, err, 0, raw)
		writeAnthropicProxyError(w, err)
		return
	}
	recordAnthropicUsage(r, options, apiKey, result.AccountID, result.Model, false, true, http.StatusOK, started, result.Usage, nil, 0, raw)
	reportChatPool(r, options, result.AccountID, true, nil, http.StatusOK, result.Model)
	content, reasoning, finish, usage, toolCalls := anthropicCompletionParts(result.Payload)
	if maxTools > 0 && len(toolCalls) > maxTools {
		toolCalls = toolCalls[:maxTools]
	}
	setAnthropicObservationHeaders(w, protocolObservation{Protocol: "anthropic",
		AccountID: result.AccountID, PreferAccount: result.PreferAccount, Failover: result.Failover,
		Fingerprint: result.Fingerprint, Accounts: result.Accounts, Prep: result.Prep, Stream: false,
	})
	if pck := strings.TrimSpace(stringValue(body["prompt_cache_key"])); pck != "" {
		w.Header().Set("X-Grok2API-Prompt-Cache-Key", pck)
	}
	payload := anthropic.Completion(messageID, result.Model, content, reasoning, finish, toolCalls, usage, allowedTools)
	// Surface sticky key in body metadata for non-streaming clients.
	if pck := strings.TrimSpace(stringValue(body["prompt_cache_key"])); pck != "" {
		payload["prompt_cache_key"] = pck
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveMessagesCountTokens(w http.ResponseWriter, r *http.Request, options Options) {
	// Local heuristic only — no pool/store required (matches Python).
	if !messageCountRouteAllowed(w, r, options) {
		return
	}
	var raw map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&raw); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	if !anthropic.HasMessagesOrSystem(raw) {
		writeAnthropicError(w, http.StatusBadRequest, "messages or system required", "invalid_request_error")
		return
	}
	writeJSON(w, http.StatusOK, anthropic.CountTokensForRequest(raw))
}

func messageRouteAllowed(w http.ResponseWriter, r *http.Request, options Options) (*auth.APIKeyRecord, bool) {
	if !options.MessagesEnabled {
		// Claude Code / Anthropic SDK parse type=error envelopes; plain detail breaks clients.
		writeAnthropicError(w, http.StatusServiceUnavailable, "Go messages route is not enabled", "api_error")
		return nil, false
	}
	if !isReady(options) {
		writeAnthropicError(w, http.StatusServiceUnavailable, readyReason(options), "api_error")
		return nil, false
	}
	var apiKey *auth.APIKeyRecord
	if options.APIKeys != nil {
		verified, err := options.APIKeys.Require(r.Context(), r)
		if err != nil {
			status := http.StatusInternalServerError
			message := err.Error()
			errType := "api_error"
			if errors.Is(err, auth.ErrInvalidAPIKey) {
				status = http.StatusUnauthorized
				message = "Invalid or missing API key"
				errType = "authentication_error"
			}
			writeAnthropicError(w, status, message, errType)
			return nil, false
		}
		apiKey = verified
	}
	if options.Store == nil && len(options.Candidates) == 0 {
		writeAnthropicError(w, http.StatusServiceUnavailable, "PostgreSQL store unavailable", "api_error")
		return nil, false
	}
	return apiKey, true
}

func messageCountRouteAllowed(w http.ResponseWriter, r *http.Request, options Options) bool {
	if !options.MessagesEnabled {
		writeAnthropicError(w, http.StatusServiceUnavailable, "Go messages route is not enabled", "api_error")
		return false
	}
	if !isReady(options) {
		writeAnthropicError(w, http.StatusServiceUnavailable, readyReason(options), "api_error")
		return false
	}
	if options.APIKeys != nil {
		if _, err := options.APIKeys.Require(r.Context(), r); err != nil {
			status := http.StatusInternalServerError
			message := err.Error()
			errType := "api_error"
			if errors.Is(err, auth.ErrInvalidAPIKey) {
				status = http.StatusUnauthorized
				message = "Invalid or missing API key"
				errType = "authentication_error"
			}
			writeAnthropicError(w, status, message, errType)
			return false
		}
	}
	return true
}

func clampCodexReasoning(raw, body map[string]any, userAgent string, enabled bool) {
	if !enabled || body == nil {
		return
	}
	tools := any(nil)
	if body != nil {
		tools = body["tools"]
	}
	if tools == nil && raw != nil {
		tools = raw["tools"]
	}
	// Proxies may strip UA; detect Codex via tools schema (exec_command/cmd).
	if !historycompact.LooksLikeCodexRequest(userAgent, tools, raw) {
		return
	}
	// Force low effort for Codex/OpenAI-native TTFT. Explicit values are overwritten
	// because xhigh/high dominate first-token latency on grok-4.5.
	body["reasoning_effort"] = "low"
	body["reasoning"] = map[string]any{"effort": "low", "summary": "auto"}
	if raw != nil {
		raw["reasoning_effort"] = "low"
		if m, ok := raw["reasoning"].(map[string]any); ok && m != nil {
			m["effort"] = "low"
			if _, ok := m["summary"]; !ok {
				m["summary"] = "auto"
			}
			raw["reasoning"] = m
		} else {
			raw["reasoning"] = map[string]any{"effort": "low", "summary": "auto"}
		}
	}
}

type shellArgKeysContextKey struct{}

func withShellArgKeys(ctx context.Context, keys map[string]string) context.Context {
	if len(keys) == 0 {
		return ctx
	}
	return context.WithValue(ctx, shellArgKeysContextKey{}, keys)
}

func shellArgKeysFrom(ctx context.Context) map[string]string {
	if ctx == nil {
		return nil
	}
	v, _ := ctx.Value(shellArgKeysContextKey{}).(map[string]string)
	return v
}

type allowedToolsContextKey struct{}

func withAllowedTools(ctx context.Context, names []string) context.Context {
	if len(names) == 0 {
		return ctx
	}
	return context.WithValue(ctx, allowedToolsContextKey{}, append([]string(nil), names...))
}

func allowedToolNamesFrom(ctx context.Context) []string {
	if ctx == nil {
		return nil
	}
	v, _ := ctx.Value(allowedToolsContextKey{}).([]string)
	return v
}

// ensureCodexShellCmdKeys forces shell-like tools to project args as "cmd" for
// Codex clients. Codex validates against its local tool schema (cmd), while we
// still send "command" upstream to Grok.
func ensureCodexShellCmdKeys(tools any, keys map[string]string) map[string]string {
	if keys == nil {
		keys = map[string]string{}
	}
	items, _ := tools.([]any)
	for _, item := range items {
		tool, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name := ""
		if fn, ok := tool["function"].(map[string]any); ok {
			name = strings.TrimSpace(fmt.Sprint(fn["name"]))
		}
		if name == "" {
			name = strings.TrimSpace(fmt.Sprint(tool["name"]))
		}
		if name == "" {
			continue
		}
		// Only shell family (include Codex exec_command / run_command / shell_command).
		if !toolcall.IsShellTool(name) {
			continue
		}
		// If client schema explicitly prefers command-only, keep it.
		pref := toolcall.PreferredShellArgKeyFromTool(tool)
		if pref == "command" {
			// Detect pure-command schema (no cmd property).
			params := any(nil)
			if fn, ok := tool["function"].(map[string]any); ok {
				params = fn["parameters"]
				if params == nil {
					params = fn["input_schema"]
				}
			}
			if params == nil {
				params = tool["parameters"]
			}
			if pm, ok := params.(map[string]any); ok {
				if props, ok := pm["properties"].(map[string]any); ok {
					_, hasCmd := props["cmd"]
					_, hasCommand := props["command"]
					if hasCommand && !hasCmd {
						keys[name] = "command"
						keys[strings.ToLower(name)] = "command"
						if nk := toolcall.NameKey(name); nk != "" {
							keys[nk] = "command"
						}
						continue
					}
				}
			}
		}
		// Default Codex shell → cmd (Codex local schema field name).
		keys[name] = "cmd"
		keys[strings.ToLower(name)] = "cmd"
		if nk := toolcall.NameKey(name); nk != "" {
			keys[nk] = "cmd"
		}
	}
	return keys
}

func serveResponses(w http.ResponseWriter, r *http.Request, options Options) {
	if !options.ResponsesEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go responses route is not enabled"})
		return
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return
	}
	var apiKey *auth.APIKeyRecord
	if options.APIKeys != nil {
		verified, err := options.APIKeys.Require(r.Context(), r)
		if err != nil {
			status := http.StatusInternalServerError
			message := err.Error()
			if errors.Is(err, auth.ErrInvalidAPIKey) {
				status = http.StatusUnauthorized
				message = "Invalid or missing API key"
			}
			writeJSON(w, status, map[string]any{"detail": message})
			return
		}
		apiKey = verified
	}
	if options.Store == nil && len(options.Candidates) == 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var raw map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&raw); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	stream, _ := raw["stream"].(bool)
	model := modelCatalog(options).Resolve(stringValue(raw["model"]))
	body := responses.BuildChatBody(raw, model)
	// Codex shell schema uses "cmd"; remember preferred keys from the *client* tools
	// before sanitize rewrites them to "command" for upstream Grok.
	keys := toolcall.ShellArgKeyMap(raw["tools"])
	// Always force shell-family tools onto "cmd". Codex validates local schema with
	// "cmd", not "command". Pure OpenAI command-only schemas are preserved by
	// ensureCodexShellCmdKeys when properties.command is exclusive and no cmd prop.
	// Do this even without UA detection — many proxies strip User-Agent.
	keys = ensureCodexShellCmdKeys(raw["tools"], keys)
	// If shell map still empty but request looks like Codex (UA or tools schema), seed common shell names.
	if len(keys) == 0 {
		var tools any
		if raw != nil {
			tools = raw["tools"]
		}
		if historycompact.LooksLikeCodexRequest(r.UserAgent(), tools, raw) {
			for _, name := range []string{
				"shell", "Shell", "bash", "Bash",
				"local_shell", "localShell", "exec", "run",
				"exec_command", "exec-command", "ExecCommand",
				"run_command", "shell_command", "ShellCommand",
			} {
				keys[name] = "cmd"
				keys[strings.ToLower(name)] = "cmd"
				keys[toolcall.NameKey(name)] = "cmd"
			}
		}
	}
	if len(keys) > 0 {
		r = r.WithContext(withShellArgKeys(r.Context(), keys))
	}
	clampCodexReasoning(raw, body, r.UserAgent(), options.runtimeConfig().CodexForceReasoningLow)
	messages, _ := body["messages"].([]map[string]any)
	if len(messages) == 0 {
		writeOpenAIError(w, http.StatusBadRequest, "input must contain at least one message", "invalid_request_error")
		return
	}
	chatReq := proxy.ChatRequest{Model: model, Stream: stream, Raw: body, UserAgent: r.UserAgent()}
	// Sticky keys for Codex multi-turn: prompt_cache_key first, then previous_response_id chain.
	// CRITICAL: never mint a *new* pck each turn from previous_response_id — that kills upstream
	// prompt cache. Always recover the same pck that produced the previous response when possible.
	pck := strings.TrimSpace(stringValue(raw["prompt_cache_key"]))
	if pck == "" {
		if meta, _ := raw["metadata"].(map[string]any); meta != nil {
			pck = strings.TrimSpace(stringValue(meta["prompt_cache_key"]))
		}
	}
	prevResp := strings.TrimSpace(stringValue(raw["previous_response_id"]))
	stickyFromPrev := ""
	if prevResp != "" {
		if acc, recoveredPCK := responseAffinityLookup(r.Context(), options, prevResp); acc != "" || recoveredPCK != "" {
			stickyFromPrev = strings.TrimSpace(acc)
			if pck == "" && recoveredPCK != "" {
				pck = recoveredPCK
			}
		}
	}
	if pck == "" {
		// Only mint a new session key for a fresh conversation root.
		// Prefer stable client session/conv headers over previous_response_id so
		// multi-turn without client pck still keeps one upstream cache key.
		if sid := strings.TrimSpace(r.Header.Get("X-Session-Id")); sid != "" {
			pck = "session:" + sid
		} else if cid := strings.TrimSpace(r.Header.Get("X-Grok-Conv-Id")); cid != "" {
			pck = "conv:" + cid
		} else if prevResp != "" {
			// Recovery missed (cold process / redis miss). Last resort only.
			pck = "respchain:" + prevResp
		}
	}
	if pck != "" {
		if chatReq.Raw == nil {
			chatReq.Raw = map[string]any{}
		}
		chatReq.Raw["prompt_cache_key"] = pck
		body["prompt_cache_key"] = pck
	}
	if prevResp != "" {
		if chatReq.Raw == nil {
			chatReq.Raw = map[string]any{}
		}
		chatReq.Raw["previous_response_id"] = prevResp
	}
	candidates, err := listCandidatesForRequest(r.Context(), options, chatReq, r.Header)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}
	// Hard-pin recovered previous_response account even when fingerprint lookup missed.
	if stickyFromPrev != "" {
		candidates = ensureStickyCandidate(r.Context(), options, candidates, stickyFromPrev)
		for i := range candidates {
			if candidates[i].ID == stickyFromPrev {
				if i > 0 {
					cand := candidates[i]
					copy(candidates[1:i+1], candidates[0:i])
					candidates[0] = cand
				}
				candidates[0].RequestCount -= 1_000_000_000
				break
			}
		}
	}
	service := newChatService(options)
	started := time.Now()
	responseID := responses.NewResponseID()
	respPolicy := historycompact.ResolveOutboundToolPolicy(
		"openai_responses",
		r.UserAgent(),
		options.runtimeConfig().OutboundMaxTools,
		options.runtimeConfig().OutboundMaxToolsOpenAI,
		options.runtimeConfig().OutboundMaxToolsResponsesNative,
		options.runtimeConfig().OutboundToolGap,
		options.runtimeConfig().OutboundToolGapNative,
	)
	if stream {
		// Codex TTFT: open SSE envelope immediately so the client is not blocked on
		// account pick + upstream first-byte. Empty/failover still happens before any
		// model payload is emitted by LiveStreamer.Start being called once only.
		flusher, canFlush := w.(http.Flusher)
		if !canFlush {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": "streaming is not supported by this response writer"})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-transform")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.Header().Set("X-Grok2API-Protocol", "openai_responses")
		if pck != "" {
			w.Header().Set("X-Grok2API-Prompt-Cache-Key", pck)
		}
		w.WriteHeader(http.StatusOK)
		early := responses.NewLiveStreamerWithMaxTools(responseID, model, allowedResponsesToolNames(body), respPolicy.MaxTools)
		early.SetShellArgKeys(shellArgKeysFrom(r.Context()))
		for _, frame := range early.Start() {
			_, _ = w.Write([]byte(frame))
		}
		if canFlush {
			flusher.Flush()
		}
		opened, err := service.OpenStreamWithResult(r.Context(), chatReq, candidates, resolvePickMode(options))
		if err != nil {
			// Emit failed terminal on already-open stream.
			for _, frame := range early.Fail(err.Error(), "server_error") {
				_, _ = w.Write([]byte(frame))
			}
			if canFlush {
				flusher.Flush()
			}
			// Per-account free-usage / 额度用完 already kicked via FailureReporter.
			recordResponsesUsage(r, options, apiKey, opened.AccountID, model, true, false, http.StatusBadGateway, started, nil, err, 0, raw)
			return
		}
		defer opened.Body.Close()
		defer releaseServerPick(options, opened.AccountID)
		// Bind this response_id so next Codex turn with previous_response_id sticks.
		bindResponseAffinity(r.Context(), options, responseID, opened.AccountID, opened.Model, pck)
		req := r
		if options.runtimeConfig().SSEKeepalive > 0 {
			req = r.WithContext(withAnthropicKeepalive(r.Context(), options.runtimeConfig().SSEKeepalive))
		}
		if respPolicy.ToolGap > 0 {
			req = req.WithContext(withOutboundToolGap(req.Context(), respPolicy.ToolGap))
		}
		setProtocolObservationHeaders(w, protocolObservation{
			Protocol: "openai_responses", AccountID: opened.AccountID, PreferAccount: opened.PreferAccount,
			Failover: opened.Failover, Fingerprint: opened.Fingerprint, Accounts: opened.Accounts, Prep: opened.Prep,
		})
		usage, firstTokenMS, err := streamOpenAIResponsesContinue(w, req, opened.Body, early, effectiveResponsesKeepalive(optionsFromRequest(req).Keepalive, len(allowedResponsesToolNames(body)) > 0), respPolicy.MaxTools)
		// Client cancel after headers is soft-ok only when the stream function
		// already soft-closed a real payload (returns nil). Empty/tool-vanished
		// returns an explicit empty error → not ok.
		ok := err == nil
		if !ok && errors.Is(err, r.Context().Err()) {
			ok = true
		}
		status := http.StatusOK
		if !ok {
			status = http.StatusBadGateway
		}
		// Hollow / half-open function_call streams return empty error from the
		// stream function (ClientDeliveryOK). Zero usage tokens alone are not a
		// failure — xAI often omits usage on short tool turns.
		recordResponsesUsage(r, options, apiKey, opened.AccountID, opened.Model, true, ok, status, started, usage, err, firstTokenMS, raw)
		reportChatPool(r, options, opened.AccountID, ok, err, status, opened.Model)
		return
	}
	chatReq.Stream = false
	result, err := service.CompleteWithResult(r.Context(), chatReq, candidates, resolvePickMode(options))
	if result.AccountID != "" {
		defer releaseServerPick(options, result.AccountID)
	}
	if err != nil {
		// Per-account free-usage / 额度用完 already kicked via FailureReporter.
		recordResponsesUsage(r, options, apiKey, result.AccountID, model, false, false, http.StatusBadGateway, started, nil, err, 0, raw)
		writeOpenAIProxyError(w, err)
		return
	}
	recordResponsesUsage(r, options, apiKey, result.AccountID, result.Model, false, true, http.StatusOK, started, result.Usage, nil, 0, raw)
	reportChatPool(r, options, result.AccountID, true, nil, http.StatusOK, result.Model)
	content, reasoning, _, _, toolCalls := anthropicCompletionParts(result.Payload)
	bindResponseAffinity(r.Context(), options, responseID, result.AccountID, result.Model, pck)
	setProtocolObservationHeaders(w, protocolObservation{
		Protocol: "openai_responses", AccountID: result.AccountID, PreferAccount: result.PreferAccount,
		Failover: result.Failover, Fingerprint: result.Fingerprint, Accounts: result.Accounts, Prep: result.Prep,
	})
	if pck != "" {
		w.Header().Set("X-Grok2API-Prompt-Cache-Key", pck)
	}
	writeJSON(w, http.StatusOK, responses.BuildObject(responseID, result.Model, content, reasoning, responseToolCalls(toolCalls, shellArgKeysFrom(r.Context())), usageMap(result.Usage), time.Now().Unix(), stringValue(raw["previous_response_id"]), metadataMap(raw["metadata"])))
}

// effectiveResponsesKeepalive tightens SSE keepalive for tool-heavy Codex turns
// so reverse proxies (~30–60s idle) do not cut the stream while incomplete
// function_call args are still buffering (upstream not idle → ReadSSEWithIdle
// would not fire keepalives otherwise — we force client keepalives from Feed).
func effectiveResponsesKeepalive(base time.Duration, toolsRequested bool) time.Duration {
	if base <= 0 {
		base = 4 * time.Second
	}
	if toolsRequested && base > 3*time.Second {
		return 3 * time.Second
	}
	// Even without declared tools, keep a sane upper bound for Codex multi-minute
	// tool loops that declare tools mid-session via previous_response.
	if base > 5*time.Second {
		return 5 * time.Second
	}
	return base
}

// responsesKeepaliveFrame is a pure SSE comment plus an OpenAI-compatible ping
// event so both dumb reverse proxies and Codex-style clients stay warm.
func responsesKeepaliveFrame() string {
	return ": keepalive\n\nevent: response.ping\ndata: {\"type\":\"response.ping\"}\n\n"
}

func streamOpenAIResponses(w http.ResponseWriter, r *http.Request, body io.Reader, responseID, model string, allowed []string, keepalive time.Duration, maxTools int) (map[string]any, int, error) {
	if _, ok := w.(http.Flusher); !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": "streaming is not supported by this response writer"})
		return nil, 0, errors.New("streaming is not supported by this response writer")
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("X-Grok2API-Protocol", "openai_responses")
	w.WriteHeader(http.StatusOK)
	if maxTools < 0 {
		maxTools = 0
	}
	streamer := responses.NewLiveStreamerWithMaxTools(responseID, model, allowed, maxTools)
	streamer.SetShellArgKeys(shellArgKeysFrom(r.Context()))
	return runOpenAIResponsesStream(w, r, body, streamer, keepalive, maxTools, false, len(allowed) > 0)
}

func responsesToolDeltas(delta proxy.ChatDelta) []responses.ToolDelta {
	chatDeltas := delta.AnthropicToolDeltas()
	out := make([]responses.ToolDelta, 0, len(chatDeltas))
	for _, item := range chatDeltas {
		out = append(out, responses.ToolDelta{Index: item.Index, ID: item.ID, Name: item.Name, Arguments: item.Arguments})
	}
	return out
}

func responsesUsageFromOpenAI(usage map[string]any) responses.Usage {
	prompt, completion, total, cacheRead, cacheCreate, reasoning := postgres.UsageFromOpenAI(usage)
	return responses.Usage{InputTokens: int(prompt), OutputTokens: int(completion), TotalTokens: int(total), CachedTokens: int(cacheRead), CacheCreationTokens: int(cacheCreate), ReasoningTokens: int(reasoning)}
}

func allowedResponsesToolNames(body map[string]any) []string {
	return allowedAnthropicToolNames(body)
}

// streamOpenAIResponsesContinue continues an already-opened Responses SSE after early envelope.
// streamOpenAIResponsesContinue continues an already-opened Responses SSE after early envelope.
func streamOpenAIResponsesContinue(w http.ResponseWriter, r *http.Request, body io.Reader, streamer *responses.LiveStreamer, keepalive time.Duration, maxTools int) (map[string]any, int, error) {
	toolsRequested := streamer != nil && (streamer.HasPendingTools() || streamer.HasClientPayload())
	return runOpenAIResponsesStream(w, r, body, streamer, keepalive, maxTools, true, toolsRequested)
}

type responseAffinityStore interface {
	BindResponseAccount(ctx context.Context, responseID, accountID, promptCacheKey string) error
	GetResponseAccount(ctx context.Context, responseID string) (accountID, promptCacheKey string, err error)
}

func responseAffinityLookup(ctx context.Context, options Options, previousResponseID string) (accountID, promptCacheKey string) {
	previousResponseID = strings.TrimSpace(previousResponseID)
	if previousResponseID == "" || options.AffinityStore == nil {
		return "", ""
	}
	// Prefer dedicated response_id map (account + recovered prompt_cache_key).
	if store, ok := options.AffinityStore.(responseAffinityStore); ok {
		acc, pck, err := store.GetResponseAccount(ctx, previousResponseID)
		if err == nil && strings.TrimSpace(acc) != "" {
			return strings.TrimSpace(acc), strings.TrimSpace(pck)
		}
	}
	// Fallback: fingerprint bindings written by bindResponseAffinity (no pck recovery).
	if id, err := options.AffinityStore.GetAffinity(ctx, "chat:previous_response_id:"+previousResponseID); err == nil {
		if acc := strings.TrimSpace(id); acc != "" {
			return acc, ""
		}
	}
	return "", ""
}

func bindResponseAffinity(ctx context.Context, options Options, responseID, accountID, model, promptCacheKey string) {
	responseID = strings.TrimSpace(responseID)
	accountID = strings.TrimSpace(accountID)
	model = strings.TrimSpace(model)
	promptCacheKey = strings.TrimSpace(promptCacheKey)
	if responseID == "" || accountID == "" || options.AffinityStore == nil {
		return
	}
	// Durable sticky for next turn:
	// 1) response_id -> (account, pck)  [local cache + redis]
	// 2) pck fingerprints -> account
	// 3) previous_response_id fingerprints -> account
	// Adapter already does optimistic local cache + one background Redis write.
	// Do NOT wrap another short-timeout goroutine here (that caused cache miss races).
	if store, ok := options.AffinityStore.(responseAffinityStore); ok {
		_ = store.BindResponseAccount(ctx, responseID, accountID, promptCacheKey)
	}
	if promptCacheKey != "" {
		_ = options.AffinityStore.BindAffinity(ctx, "chat:prompt_cache_key:"+promptCacheKey, accountID)
		if model != "" {
			_ = options.AffinityStore.BindAffinity(ctx, "chat:"+model+":prompt_cache_key:"+promptCacheKey, accountID)
		}
	}
	_ = options.AffinityStore.BindAffinity(ctx, "chat:previous_response_id:"+responseID, accountID)
	if model != "" {
		_ = options.AffinityStore.BindAffinity(ctx, "chat:"+model+":previous_response_id:"+responseID, accountID)
	}
}

func recordResponsesUsage(r *http.Request, options Options, apiKey *auth.APIKeyRecord, accountID, model string, stream bool, ok bool, status int, started time.Time, usage any, cause error, ttftMS int, requestBody map[string]any) {
	prompt, completion, total, cacheRead, cacheCreate, reasoning := postgres.UsageFromOpenAI(usage)
	streamValue := stream
	var apiKeyID string
	if apiKey != nil {
		apiKeyID = apiKey.ID
	}
	var errText string
	if cause != nil {
		errText = cause.Error()
	}
	latency := int(time.Since(started).Milliseconds())
	var ttftPtr *int
	if ttftMS > 0 {
		v := ttftMS
		ttftPtr = &v
	}
	detail := map[string]any{"route": "go_responses"}
	if effort := extractReasoningEffort(requestBody); effort != "" {
		detail["reasoning_effort"] = effort
		detail["thinking_intensity"] = effort
	}
	if ttftMS > 0 {
		detail["ttft_ms"] = ttftMS
	}
	detail["latency_ms"] = latency
	// Fire-and-forget with longer timeout - usage recording should not block response
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if options.Store != nil {
			if _, _, err := options.Store.RecordUsage(ctx, postgres.UsageRecord{
				RequestID:           requestID(r),
				Implementation:      "go",
				APIKeyID:            apiKeyID,
				AccountID:           accountID,
				Model:               model,
				Protocol:            "openai_responses",
				Path:                r.URL.Path,
				Stream:              &streamValue,
				OK:                  ok,
				PromptTokens:        prompt,
				CompletionTokens:    completion,
				TotalTokens:         total,
				CacheReadTokens:     cacheRead,
				CacheCreationTokens: cacheCreate,
				ReasoningTokens:     reasoning,
				ClientIP:            clientIP(r),
				UserAgent:           r.UserAgent(),
				StatusCode:          &status,
				LatencyMS:           &latency,
				TTFTMS:              ttftPtr,
				Error:               errText,
				Detail:              detail,
			}); err != nil {
				slog.Warn("record usage failed", "error", err, "account_id", accountID, "model", model, "ok", ok)
			}
		}
		recordRedisUsage(options, apiKeyID, accountID, model, prompt, completion, total, cacheRead, ok)
	}()
}

func responseToolCalls(calls []anthropic.ToolCall, shellArgKeys ...map[string]string) []map[string]any {
	keys := map[string]string{}
	if len(shellArgKeys) > 0 && shellArgKeys[0] != nil {
		keys = shellArgKeys[0]
	}
	out := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		name := strings.TrimSpace(call.Name)
		args := call.Arguments
		// Project shell args to client schema (Codex: cmd). Internal form is command.
		if toolcall.IsShellTool(name) {
			preferred := "cmd"
			if v := strings.TrimSpace(keys[name]); v != "" {
				preferred = v
			} else if v := strings.TrimSpace(keys[strings.ToLower(name)]); v != "" {
				preferred = v
			} else if nk := toolcall.NameKey(name); nk != "" {
				if v := strings.TrimSpace(keys[nk]); v != "" {
					preferred = v
				}
			}
			args = toolcall.ProjectShellArgsForClient(args, name, preferred)
		}
		out = append(out, map[string]any{
			"id":   call.ID,
			"type": "function",
			"function": map[string]any{
				"name":      name,
				"arguments": args,
			},
		})
	}
	return out
}

func usageMap(value any) map[string]any {
	usage, _ := value.(map[string]any)
	if usage == nil {
		return map[string]any{}
	}
	return usage
}

func metadataMap(value any) map[string]any {
	metadata, _ := value.(map[string]any)
	return metadata
}

type protocolObservation struct {
	Protocol      string
	AccountID     string
	PreferAccount string
	Failover      bool
	Fingerprint   string
	Accounts      int
	Prep          proxy.BodyPrepStats
	Stream        bool
}

func setAnthropicObservationHeaders(w http.ResponseWriter, obs protocolObservation) {
	if obs.Protocol == "" {
		obs.Protocol = "anthropic"
	}
	setProtocolObservationHeaders(w, obs)
}

func setProtocolObservationHeaders(w http.ResponseWriter, obs protocolObservation) {
	if obs.Protocol == "" {
		obs.Protocol = "go"
	}
	w.Header().Set("X-Grok2API-Protocol", obs.Protocol)
	if obs.Accounts > 0 {
		w.Header().Set("X-Grok2API-Accounts", strconv.Itoa(obs.Accounts))
	}
	if obs.PreferAccount != "" {
		w.Header().Set("X-Grok2API-Affinity", "1")
	} else {
		w.Header().Set("X-Grok2API-Affinity", "0")
	}
	if obs.Failover {
		w.Header().Set("X-Grok2API-Affinity-Rebind", "1")
	}
	if obs.Fingerprint != "" {
		w.Header().Set("X-Grok2API-Conversation-Fp", obs.Fingerprint)
	}
	if obs.AccountID != "" {
		w.Header().Set("X-Grok2API-Account", obs.AccountID)
	}
	if compact := obs.Prep.Compact; compact != nil {
		if truthyAny(compact["applied"]) {
			w.Header().Set("X-Grok2API-History-Compact", "1")
		} else {
			w.Header().Set("X-Grok2API-History-Compact", "0")
		}
		if v, ok := compact["before_chars"]; ok {
			w.Header().Set("X-Grok2API-History-Before", fmt.Sprint(v))
		}
		if v, ok := compact["after_chars"]; ok {
			w.Header().Set("X-Grok2API-History-After", fmt.Sprint(v))
		}
		if v, ok := compact["tool_rounds"]; ok {
			w.Header().Set("X-Grok2API-History-Tool-Rounds", fmt.Sprint(v))
		}
		if truthyAny(compact["prefix_stable"]) {
			w.Header().Set("X-Grok2API-History-Prefix-Stable", "1")
		}
		if truthyAny(compact["auto"]) {
			w.Header().Set("X-Grok2API-History-Auto", "1")
		}
	}
	if stabilize := obs.Prep.Stabilize; stabilize != nil {
		w.Header().Set("X-Grok2API-Prompt-Stable", "1")
		w.Header().Set("X-Grok2API-Prompt-Stable-Messages", fmt.Sprint(stabilize["messages_stabilized"]))
		w.Header().Set("X-Grok2API-Prompt-Stable-Tools", fmt.Sprint(stabilize["tools_stabilized"]))
	} else {
		w.Header().Set("X-Grok2API-Prompt-Stable", "0")
	}
}

func truthyAny(value any) bool {
	switch v := value.(type) {
	case bool:
		return v
	case int:
		return v != 0
	case int64:
		return v != 0
	case float64:
		return v != 0
	case string:
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			return true
		default:
			return false
		}
	default:
		return false
	}
}

func streamAnthropicMessages(w http.ResponseWriter, r *http.Request, body io.Reader, messageID, model string, toolsRequested bool, allowed []string, maxTools int) (map[string]any, int, error) {
	return streamAnthropicMessagesWithOptions(w, r, body, messageID, model, toolsRequested, allowed, maxTools, optionsFromRequest(r))
}

type anthropicStreamOptions struct {
	Keepalive time.Duration
}

func optionsFromRequest(r *http.Request) anthropicStreamOptions {
	// Default matches Python SSE_KEEPALIVE_INTERVAL (~4s). Tests can override via context value.
	keepalive := 4 * time.Second
	if r != nil {
		if value := r.Context().Value(anthropicKeepaliveContextKey{}); value != nil {
			if d, ok := value.(time.Duration); ok {
				keepalive = d
			}
		}
	}
	return anthropicStreamOptions{Keepalive: keepalive}
}

type anthropicKeepaliveContextKey struct{}

func withAnthropicKeepalive(ctx context.Context, d time.Duration) context.Context {
	return context.WithValue(ctx, anthropicKeepaliveContextKey{}, d)
}

type outboundToolGapContextKey struct{}

func withOutboundToolGap(ctx context.Context, d time.Duration) context.Context {
	return context.WithValue(ctx, outboundToolGapContextKey{}, d)
}

func outboundToolGapFrom(ctx context.Context) time.Duration {
	if ctx == nil {
		return 0
	}
	if value := ctx.Value(outboundToolGapContextKey{}); value != nil {
		if d, ok := value.(time.Duration); ok {
			return d
		}
	}
	return 0
}

func streamAnthropicMessagesWithOptions(w http.ResponseWriter, r *http.Request, body io.Reader, messageID, model string, toolsRequested bool, allowed []string, maxTools int, opts anthropicStreamOptions) (map[string]any, int, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": "streaming is not supported by this response writer"})
		return nil, 0, errors.New("streaming is not supported by this response writer")
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("X-Grok2API-Protocol", "anthropic")
	w.WriteHeader(http.StatusOK)

	started := time.Now()
	firstTokenMS := 0

	assembler := anthropic.NewStreamAssembler(messageID, model, toolsRequested, maxTools, allowed)
	// Claude Code / reverse proxies often blip once under write pressure; require
	// multiple consecutive ctx-done hits across a short span before treating as gone.
	probe := newDisconnectProbe(3, 800*time.Millisecond)
	envelopeOpen := false
	// Soft client write/ctx blips after envelope open must NOT abort ReadSSE:
	// abort drains remaining upstream tool-arg frames → force-finish half tools →
	// Claude Code "Tool use interrupted". Keep assembling; only best-effort write.
	sw := newSSEWriter(w, flusher, r.Context())
	var writeMu sync.Mutex
	// writeFrames writes one logical batch via shared sseWriter. Soft errors leave
	// LastOK=false so callers Requeue instead of Acking half-open tool_use.
	// Force is automatic once envelope is open so terminal tools can still land.
	writeFrames := func(frames []string, force bool) error {
		if len(frames) == 0 {
			return nil
		}
		writeMu.Lock()
		defer writeMu.Unlock()
		return sw.WriteStrings(frames, force || envelopeOpen || sw.SoftGone())
	}
	// ackBatchFromFrames marks only what this Write batch actually contained.
	// Blind AckEmittedTools after a text/thinking-only flush would wrongly mark
	// not-yet-written tools acked; missing Ack of message_stop leaves terminal
	// pending stuck. Both paths surface as intermittent "Tool use interrupted".
	// Each tool group is one content_block_start with content_block.type=tool_use
	// (see anthropic.event JSON: `"type":"tool_use"` once per group).
	ackBatchFromFrames := func(batch []string) {
		joined := strings.Join(batch, "")
		if strings.Contains(joined, "event: message_start") || strings.Contains(joined, `"type":"message_start"`) {
			assembler.AckMessageStart()
		}
		// Content-based tool Ack (by tool id in payload), not FIFO count. Multi-tool
		// Finish writes groups independently; soft-fail of tool0 + success of tool1
		// must not Ack tool0 via AckFirstPendingTools(1).
		if strings.Contains(joined, `"type":"tool_use"`) || strings.Contains(joined, "tool_use") {
			assembler.AckToolsInPayload(joined)
		}
		if strings.Contains(joined, "event: message_stop") || strings.Contains(joined, `"type":"message_stop"`) {
			assembler.AckTerminal()
		}
	}
	toolGap := outboundToolGapFrom(r.Context())
	toolsEmitted := 0
	// pendingSSE coalesces live thinking/text micro-deltas (Claude Code long
	// thinking turns) so we do not Flush once per token. Flushed on tool groups,
	// terminal, size, idle, or first client-visible payload (TTFT).
	pendingSSE := make([]byte, 0, textCoalesceMax*2)
	firstPayloadFlushed := false
	flushPendingSSE := func(force bool) error {
		if len(pendingSSE) == 0 {
			return nil
		}
		payload := pendingSSE
		pendingSSE = pendingSSE[:0]
		streamCoalesceFlush.Add(1)
		// Direct WriteBytes: payload is already concatenated SSE frames.
		writeMu.Lock()
		err := sw.WriteBytes(payload, force || sw.SoftGone() || envelopeOpen)
		writeMu.Unlock()
		if err != nil {
			return err
		}
		if sw.LastOK() {
			envelopeOpen = true
			firstPayloadFlushed = true
		}
		return nil
	}
	// Precompute keepalive once — Ping()+CommentKeepalive() allocate every call.
	anthropicKeepaliveFrame := anthropic.Ping() + anthropic.CommentKeepalive()
	// flushGroup writes one logical batch (text/thinking, one tool group, or
	// message_delta+stop). requeueOnSoft controls whether a soft skip/fail
	// immediately rolls assembler state back.
	//
	// Finish multi-group path MUST use requeueOnSoft=false for every group and
	// only Requeue once AFTER all groups attempt. Mid-Finish Requeue clears
	// pendingTerminal while later groups still hold the same message_stop bytes;
	// a successful terminal Write then cannot AckTerminal → outer Finish re-emits
	// tools + second message_stop → Claude Code "Tool use interrupted".
	flushGroup := func(all []string, force bool, requeueOnSoft bool) error {
		if len(all) == 0 {
			return nil
		}
		if err := writeFrames(all, force || sw.SoftGone() || envelopeOpen); err != nil {
			if requeueOnSoft && assembler.HasUnackedTools() {
				assembler.RequeueUnackedTools()
			}
			return err
		}
		envelopeOpen = true
		if sw.LastOK() {
			ackBatchFromFrames(all)
		} else if requeueOnSoft && assembler.HasUnackedTools() {
			assembler.RequeueUnackedTools()
		}
		return nil
	}
	// flushGroupRetry re-attempts the SAME frame bytes after soft write blips.
	// Never Requeue between attempts or between Finish groups — keep pending*
	// so a successful retry can Ack. Caller requeues after the full multi-group
	// emit if anything remains unacked.
	flushGroupRetry := func(all []string, force bool, attempts int) error {
		if attempts < 1 {
			attempts = 1
		}
		var lastErr error
		for i := 0; i < attempts; i++ {
			if i > 0 {
				// Brief backoff so a write-pressure blip can clear; keep total
				// Finish recovery under a few ms for local sockets.
				time.Sleep(time.Duration(i) * 2 * time.Millisecond)
			}
			lastErr = flushGroup(all, force, false)
			if lastErr != nil {
				return lastErr
			}
			if sw.LastOK() {
				return nil
			}
		}
		return lastErr
	}
	emitFrames := func(frames []string, force bool) error {
		if len(frames) == 0 {
			return nil
		}
		// Classify batch: pure thinking/text can coalesce; tools/terminal/message_start force.
		needImmediate := force
		if !needImmediate {
			for _, frame := range frames {
				if frameNeedsAnthropicImmediate(frame) {
					needImmediate = true
					break
				}
			}
		}
		if !needImmediate {
			// Pure thinking_delta / text_delta path — coalesce micro frames.
			for _, frame := range frames {
				pendingSSE = append(pendingSSE, frame...)
			}
			if !firstPayloadFlushed || len(pendingSSE) >= textCoalesceMax {
				return flushPendingSSE(true)
			}
			return nil
		}
		// Tools/terminal/start: drain coalesced thinking/text first (order: text then tools).
		if err := flushPendingSSE(true); err != nil {
			return err
		}
		// Keep each tool_use start+delta+stop in one Write+Flush. Flushing on
		// content_block_start alone leaves Claude Code with a half-open tool_use
		// block; a soft disconnect mid-group surfaces as "Tool use interrupted".
		// toolGap still applies BETWEEN tools (flush previous group first).
		//
		// Finish (message_stop present): emit tool groups and terminal SEPARATELY
		// with per-group Ack. A single giant multi-tool Write that soft-fails (or
		// short-writes) would either drop everything or Ack frames the client never
		// saw. Ordered per-tool Ack + terminal-last is what fully closes Claude Code.
		hasTerminal := false
		for _, f := range frames {
			if strings.Contains(f, "message_stop") {
				hasTerminal = true
				break
			}
		}
		// Split frames into ordered groups: non-tool prefix, each tool_use
		// start+delta+stop cluster, and trailing terminal (message_delta/stop).
		// Tool clusters start at content_block_start+tool_use.
		groups := make([][]string, 0, 4)
		cur := make([]string, 0, 4)
		flushCur := func() {
			if len(cur) == 0 {
				return
			}
			groups = append(groups, cur)
			cur = make([]string, 0, 4)
		}
		for _, frame := range frames {
			isToolStart := frameIsAnthropicToolStart(frame)
			isTerminalFrame := frameIsAnthropicTerminal(frame)
			if isToolStart {
				flushCur()
			} else if isTerminalFrame && len(cur) > 0 {
				// Do not mix tool/text frames with terminal in one Write.
				// Detect first terminal-ish frame after non-terminal content.
				joinedCur := strings.Join(cur, "")
				if strings.Contains(joinedCur, `"type":"tool_use"`) || strings.Contains(joinedCur, "content_block") ||
					strings.Contains(joinedCur, "message_start") {
					flushCur()
				}
			}
			cur = append(cur, frame)
		}
		flushCur()

		if hasTerminal {
			// Finish path: per-group retry (tools first, terminal last). Do NOT
			// requeue between groups — pending* must stay until Ack or until the
			// whole multi-group emit ends. Soft-fail of tool N leaves it unacked;
			// terminal success still Acks message_stop; outer Finish rebuilds
			// only the remaining unacked tools (toolsOnly when terminal already
			// delivered).
			var firstHard error
			for _, g := range groups {
				if err := flushGroupRetry(g, force, 3); err != nil {
					if firstHard == nil {
						firstHard = err
					}
					// Keep trying remaining groups (esp. message_stop) so Claude
					// Code can leave "running" even if one tool Write hard-failed.
				}
			}
			if assembler.HasUnackedTools() {
				// Roll back anything that never Ack'd so outer Finish can rebuild
				// complete start+delta+stop groups (and message_stop if needed).
				assembler.RequeueUnackedTools()
			}
			return firstHard
		}

		// Live mid-stream: same per-tool grouping; single attempt here — outer
		// Finish recovery re-emits anything left unacked.
		for i, g := range groups {
			// toolGap between live tool groups (not before the first).
			if toolGap > 0 && toolsEmitted > 0 {
				isToolGroup := false
				for _, f := range g {
					if strings.Contains(f, `"tool_use"`) {
						isToolGroup = true
						break
					}
				}
				if isToolGroup {
					if waitToolGap(r.Context(), toolGap) {
						sw.MarkSoftGone()
						if !envelopeOpen && !force {
							return r.Context().Err()
						}
					}
				}
			}
			// Live path: requeue on soft fail so Finish can rebuild complete groups.
			if err := flushGroup(g, force, true); err != nil {
				return err
			}
			for _, f := range g {
				if strings.Contains(f, `"tool_use"`) && strings.Contains(f, "content_block_start") {
					toolsEmitted++
					break
				}
			}
			_ = i
		}
		return nil
	}

	// Early envelope: Claude Code perceived TTFT starts when message_start lands,
	// not when the first model token arrives (important on long tool-prep turns).
	if err := emitFrames(assembler.Start(0), true); err != nil {
		return nil, 0, err
	}
	envelopeOpen = true

	var finish string
	var usage anthropic.Usage
	var openAIUsage map[string]any
	sawModel := false

	// Keepalive: prefer a slightly tighter interval for toolsRequested turns so
	// long thinking never sits near proxy idle cutoffs (~60s).
	keepalive := opts.Keepalive
	if keepalive <= 0 {
		keepalive = 4 * time.Second
	}
	if toolsRequested && keepalive > 3*time.Second {
		keepalive = 3 * time.Second
	}

	onIdle := func() error {
		// Drain coalesced thinking/text so clients are not stuck waiting for size.
		_ = flushPendingSSE(true)
		if probe.check(r.Context()) {
			sw.MarkSoftGone()
			if !envelopeOpen {
				return r.Context().Err()
			}
		}
		// Always force when holding tools/text so idle timer resets client-side even
		// under write-pressure (Claude Code multi-minute Update/Edit turns).
		force := toolsRequested || assembler.NeedsClientKeepalive() || sw.SoftGone() || envelopeOpen
		return sw.Keepalive(anthropicKeepaliveFrame, DefaultKeepaliveInterval, force)
	}

	err := grok.ReadSSEWithIdle(body, keepalive, func(event grok.Event) error {
		// After envelope is open, soft-disconnect probes must not abort mid-stream;
		// terminal frames still need to land so Claude Code can leave "running".
		if probe.check(r.Context()) {
			sw.MarkSoftGone()
			if !envelopeOpen {
				return r.Context().Err()
			}
		}
		if event.Done {
			return nil
		}
		delta, err := proxy.ParseChatDelta(event.Data)
		if err != nil {
			// Ignore malformed keep-alive / partial frames; do not count as TTFT.
			return nil
		}
		if delta.FinishReason != nil {
			finish = stringValue(delta.FinishReason)
		}
		if delta.Usage != nil {
			if raw, ok := delta.Usage.(map[string]any); ok {
				openAIUsage = raw
				prompt, completion, total, cacheRead, cacheCreate, _ := postgres.UsageFromOpenAI(raw)
				usage = anthropic.Usage{PromptTokens: int(prompt), CompletionTokens: int(completion), TotalTokens: int(total), CacheReadTokens: int(cacheRead), CacheCreationTokens: int(cacheCreate)}
			}
		}
		hasContent := strings.TrimSpace(delta.Content) != ""
		hasReasoning := strings.TrimSpace(delta.Reasoning) != ""
		hasTools := len(delta.AnthropicToolDeltas()) > 0
		if hasContent || hasReasoning || hasTools {
			sawModel = true
			// TTFT = first real model payload (not empty SSE / not message_start alone).
			if firstTokenMS == 0 {
				firstTokenMS = int(time.Since(started).Milliseconds())
				if firstTokenMS <= 0 {
					firstTokenMS = 1
				}
			}
		}
		// Live path: do NOT force-flush pure thinking/text (coalesce). Tools/start
		// inside Feed still force via needImmediate classification in emitFrames.
		// Soft write errors are still swallowed by sseWriter after envelope open.
		wroteFrames := false
		feedFrames := assembler.Feed(delta.Content, delta.Reasoning, delta.AnthropicToolDeltas())
		if len(feedFrames) > 0 {
			if err := emitFrames(feedFrames, false); err != nil {
				return err
			}
			wroteFrames = true
		}
		// Incomplete tool args / held text: throttled keepalive. Skip when we just
		// wrote real frames this tick (socket already warm).
		if assembler.NeedsClientKeepalive() && !wroteFrames && len(pendingSSE) == 0 {
			return sw.Keepalive(anthropicKeepaliveFrame, DefaultKeepaliveInterval, true)
		}
		return nil
	}, onIdle)

	// Drain any coalesced thinking/text before terminal Finish / empty checks.
	_ = flushPendingSSE(true)

	clientGone := sw.SoftGone() || probe.gone || errors.Is(err, r.Context().Err()) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || isSoftClientWriteError(err)
	// Real payload = model deltas OR assembler emitted/held content/tools.
	// Early message_start alone must NOT count as success (admin showed ok=true,
	// tokens=0, ttft=null for ~1s intermittent empties).
	hasPayload := sawModel || assembler.HasClientPayload() || assembler.HasPendingTools() || assembler.HasHeldContent()
	// Upstream read/error mid-stream AFTER the client already has content/tools must
	// soft-close the Anthropic envelope (Finish only). Emitting event:error here is
	// what Claude Code surfaces as "API Error: Server error mid-response / response
	// above may be incomplete" even when most of the turn already landed.
	upstreamMidError := err != nil && !clientGone
	if upstreamMidError && !hasPayload {
		msg, errType := anthropicErrorFromCause(err)
		_ = emitFrames(anthropic.TerminalError(msg, errType), true)
		return openAIUsage, firstTokenMS, err
	}
	if !hasPayload {
		// Soft disconnect before any model payload is still a hollow stream —
		// never soft-ok (admin ok=true tokens=0). Close envelope if open, then fail.
		empty := errors.New("Upstream returned HTTP 200 with empty model output (no content/tool_calls)")
		if clientGone {
			_ = emitFrames(assembler.Finish("stop", usage), true)
			return openAIUsage, firstTokenMS, empty
		}
		msg, errType := anthropicErrorFromCause(empty)
		_ = emitFrames(anthropic.TerminalError(msg, errType), true)
		return openAIUsage, firstTokenMS, empty
	}
	if finish == "" {
		finish = "stop"
	}
	// Always try terminal frames after any payload (including write-error soft disconnect
	// and mid-stream upstream errors) so Claude Code never hangs on a half-open
	// tool_use as "Tool use interrupted" / "Server error mid-response".
	// Finish requeues unacked tools first; emitFrames writes tool groups then
	// message_delta/stop as separate groups with per-group retry. Soft write of a
	// group Requeues (clears pending) — NeedsFinishRetry still sees
	// !TerminalDelivered / pending tools and forces another Finish.
	if termErr := emitFrames(assembler.Finish(finish, usage), true); termErr != nil && !clientGone && !upstreamMidError {
		return openAIUsage, firstTokenMS, termErr
	}
	// Soft-fail recovery: more Finish rebuilds (tools + message_stop). Each rebuild
	// re-emits only unacked/pending work; Ack'd tools are skipped. Requeue also
	// UnackTerminal when tools need re-emit so tool_use never follows message_stop
	// ("Tool use interrupted").
	for attempt := 0; attempt < 4 && assembler.NeedsFinishRetry(); attempt++ {
		_ = emitFrames(assembler.Finish(finish, usage), true)
	}
	// After Finish, incomplete-only tools (never started) + no text is empty —
	// even if sawModel was true from upstream tool deltas. That was the intermittent
	// ok=true tokens=0 TTFT>0 path Claude Code reports as "Tool use interrupted".
	if !assembler.HasClientPayload() {
		empty := errors.New("Upstream returned HTTP 200 with empty model output (no content/tool_calls)")
		if !assembler.TerminalDelivered() {
			_ = emitFrames(anthropic.TerminalError(empty.Error(), "api_error"), true)
		}
		return openAIUsage, firstTokenMS, empty
	}
	// Recovery exhausted with half-open tools / no terminal: fail (not soft-ok).
	// clientGone after undelivered tools is still a failure — Claude Code shows
	// "Tool use interrupted" and admin must not record ok=true tokens=0.
	if !assembler.ClientDeliveryOK() {
		empty := errors.New("Upstream returned HTTP 200 with empty model output (no content/tool_calls)")
		if !assembler.TerminalDelivered() {
			_ = emitFrames(anthropic.TerminalError(empty.Error(), "api_error"), true)
		}
		if upstreamMidError && err != nil {
			return openAIUsage, firstTokenMS, err
		}
		return openAIUsage, firstTokenMS, empty
	}
	if clientGone || upstreamMidError {
		// Soft terminal: client left, or upstream dropped after FULL delivery.
		// Do not return the raw upstream err — stream already closed cleanly.
		return openAIUsage, firstTokenMS, nil
	}
	return openAIUsage, firstTokenMS, err
}

// isSoftClientWriteError reports client-side stream aborts that should still run
// Finish/Complete so Claude Code / Codex leave "running" instead of hanging as
// "Tool use interrupted" on a half-open tool block.
func isSoftClientWriteError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, needle := range []string{
		"broken pipe",
		"connection reset",
		"connection refused",
		"use of closed network connection",
		"stream closed",
		"client disconnected",
		"i/o timeout",
		"write: connection",
		"short write",
		"wsasend",
	} {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

type disconnectProbe struct {
	hitsNeeded int
	minSpan    time.Duration
	hits       int
	firstHit   time.Time
	gone       bool
}

func newDisconnectProbe(hitsNeeded int, minSpan time.Duration) *disconnectProbe {
	if hitsNeeded < 1 {
		hitsNeeded = 1
	}
	return &disconnectProbe{hitsNeeded: hitsNeeded, minSpan: minSpan}
}

func (p *disconnectProbe) check(ctx context.Context) bool {
	if p == nil {
		return false
	}
	if p.gone {
		return true
	}
	select {
	case <-ctx.Done():
		now := time.Now()
		if p.hits == 0 {
			p.firstHit = now
		}
		p.hits++
		if p.hits >= p.hitsNeeded && (p.minSpan <= 0 || now.Sub(p.firstHit) >= p.minSpan) {
			p.gone = true
			return true
		}
		return false
	default:
		p.hits = 0
		p.firstHit = time.Time{}
		return false
	}
}

func recordAnthropicUsage(r *http.Request, options Options, apiKey *auth.APIKeyRecord, accountID, model string, stream bool, ok bool, status int, started time.Time, usage any, cause error, ttftMS int, requestBody map[string]any) {
	prompt, completion, total, cacheRead, cacheCreate, reasoning := postgres.UsageFromOpenAI(usage)
	streamValue := stream
	var apiKeyID string
	if apiKey != nil {
		apiKeyID = apiKey.ID
	}
	var errText string
	if cause != nil {
		errText = cause.Error()
	}
	latency := int(time.Since(started).Milliseconds())
	var ttftPtr *int
	if ttftMS > 0 {
		v := ttftMS
		ttftPtr = &v
	}
	// Fire-and-forget with longer timeout - usage recording should not block response
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if options.Store != nil {
			if _, _, err := options.Store.RecordUsage(ctx, postgres.UsageRecord{
				RequestID:           requestID(r),
				Implementation:      "go",
				APIKeyID:            apiKeyID,
				AccountID:           accountID,
				Model:               model,
				Protocol:            "anthropic",
				Path:                r.URL.Path,
				Stream:              &streamValue,
				OK:                  ok,
				PromptTokens:        prompt,
				CompletionTokens:    completion,
				TotalTokens:         total,
				CacheReadTokens:     cacheRead,
				CacheCreationTokens: cacheCreate,
				ReasoningTokens:     reasoning,
				ClientIP:            clientIP(r),
				UserAgent:           r.UserAgent(),
				StatusCode:          &status,
				LatencyMS:           &latency,
				TTFTMS:              ttftPtr,
				Error:               errText,
				Detail: func() map[string]any {
					d := usageDetail("go_messages", requestBody, ttftMS, latency)
					if !ok && (prompt+completion+total) == 0 {
						d["empty_payload"] = true
						if cause != nil {
							d["message"] = cause.Error()
						}
					}
					if ttftMS <= 0 {
						d["ttft_missing"] = true
					}
					return d
				}(),
			}); err != nil {
				slog.Warn("record usage failed", "error", err, "account_id", accountID, "model", model, "ok", ok)
			}
		}
		recordRedisUsage(options, apiKeyID, accountID, model, prompt, completion, total, cacheRead, ok)
	}()
}

func anthropicCompletionParts(payload map[string]any) (content, reasoning, finish string, usage anthropic.Usage, calls []anthropic.ToolCall) {
	usage = anthropic.Usage{}
	if rawUsage, ok := payload["usage"].(map[string]any); ok {
		prompt, completion, total, cacheRead, cacheCreate, _ := postgres.UsageFromOpenAI(rawUsage)
		usage = anthropic.Usage{PromptTokens: int(prompt), CompletionTokens: int(completion), TotalTokens: int(total), CacheReadTokens: int(cacheRead), CacheCreationTokens: int(cacheCreate)}
	}
	choices, _ := payload["choices"].([]map[string]any)
	if len(choices) == 0 {
		return "", "", "stop", usage, nil
	}
	finish = stringValue(choices[0]["finish_reason"])
	message, _ := choices[0]["message"].(map[string]any)
	content = stringValue(message["content"])
	reasoning = firstNonEmpty(stringValue(message["reasoning_content"]), stringValue(message["reasoning"]))
	if items, ok := message["tool_calls"].([]map[string]any); ok {
		for _, item := range items {
			fn, _ := item["function"].(map[string]any)
			calls = append(calls, anthropic.ToolCall{ID: stringValue(item["id"]), Name: stringValue(fn["name"]), Arguments: stringValue(fn["arguments"])})
		}
	} else if itemsAny, ok := message["tool_calls"].([]any); ok {
		// Some decode paths yield []any rather than []map[string]any.
		for _, rawItem := range itemsAny {
			item, _ := rawItem.(map[string]any)
			if item == nil {
				continue
			}
			fn, _ := item["function"].(map[string]any)
			calls = append(calls, anthropic.ToolCall{ID: stringValue(item["id"]), Name: stringValue(fn["name"]), Arguments: stringValue(fn["arguments"])})
		}
	}
	if fn, ok := message["function_call"].(map[string]any); ok {
		calls = append(calls, anthropic.ToolCall{Name: stringValue(fn["name"]), Arguments: stringValue(fn["arguments"])})
	}
	return content, reasoning, finish, usage, calls
}

func allowedAnthropicToolNames(body map[string]any) []string {
	if body == nil {
		return nil
	}
	items, _ := body["tools"].([]any)
	if items == nil {
		// Some clients send tools as typed maps after decode.
		if typed, ok := body["tools"].([]map[string]any); ok {
			items = make([]any, len(typed))
			for i, t := range typed {
				items[i] = t
			}
		}
	}
	out := make([]string, 0, len(items))
	seen := map[string]bool{}
	for _, item := range items {
		tool, _ := item.(map[string]any)
		if tool == nil {
			continue
		}
		// OpenAI: {"type":"function","function":{"name":"Edit",...}}
		// Anthropic: {"name":"Edit","input_schema":{...}}
		fn, _ := tool["function"].(map[string]any)
		name := stringValue(fn["name"])
		if name == "" {
			name = stringValue(tool["name"])
		}
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return out
}

func newAnthropicMessageID() string {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err != nil {
		return "msg_go_" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	return "msg_" + hex.EncodeToString(buf)
}

func positiveNumber(value any) bool {
	switch v := value.(type) {
	case int:
		return v >= 1
	case int64:
		return v >= 1
	case float64:
		return v >= 1
	case json.Number:
		n, err := v.Int64()
		return err == nil && n >= 1
	default:
		return false
	}
}

func stringValue(value any) string {
	text, _ := value.(string)
	return strings.TrimSpace(text)
}

// writeAnthropicProxyError maps upstream/proxy failures onto Anthropic JSON errors
// with useful status codes (429/401/503) instead of always 502 + raw Error() text.
func writeAnthropicProxyError(w http.ResponseWriter, err error) {
	status := http.StatusBadGateway
	message := "request failed"
	errorType := "api_error"
	if err != nil {
		message = strings.TrimSpace(err.Error())
	}
	if errors.Is(err, pool.ErrNoEligibleAccounts) {
		status = http.StatusServiceUnavailable
		message = "No eligible accounts available. All accounts may be in cooldown or disabled."
		errorType = "api_error"
		writeAnthropicError(w, status, message, errorType)
		return
	}
	var upstream *grok.UpstreamError
	if errors.As(err, &upstream) && upstream != nil {
		status = mapUpstreamStatusToAnthropic(upstream.Status)
		if strings.TrimSpace(upstream.Body) != "" {
			message = preferAnthropicErrorBody(upstream.Body)
		}
		errorType = anthropicErrorTypeForStatus(status)
		writeAnthropicError(w, status, message, errorType)
		return
	}
	msg, typ := anthropicErrorFromCause(err)
	if unwrappedStatus, unwrappedBody, ok := unwrapAnthropicUpstreamMessage(message); ok {
		if unwrappedStatus > 0 {
			status = mapUpstreamStatusToAnthropic(unwrappedStatus)
		}
		if strings.TrimSpace(unwrappedBody) != "" {
			msg = unwrappedBody
		}
		typ = anthropicErrorTypeForStatus(status)
	}
	writeAnthropicError(w, status, msg, typ)
}

func writeAnthropicError(w http.ResponseWriter, status int, message, errorType string) {
	// Prefer structured upstream body/status when callers pass cause.Error() wrappers
	// like "upstream status 429: {...}" so Claude clients see the real quota/rate text.
	if status <= 0 {
		status = http.StatusBadGateway
	}
	if unwrappedStatus, unwrappedBody, ok := unwrapAnthropicUpstreamMessage(message); ok {
		if unwrappedStatus > 0 {
			status = mapUpstreamStatusToAnthropic(unwrappedStatus)
		}
		if strings.TrimSpace(unwrappedBody) != "" {
			message = unwrappedBody
		}
		if errorType == "" || errorType == "api_error" {
			errorType = anthropicErrorTypeForStatus(status)
		}
	}
	if errorType == "" {
		errorType = anthropicErrorTypeForStatus(status)
	}
	if strings.TrimSpace(message) == "" {
		message = "request failed"
	}
	// Anthropic wire shape: { "type":"error", "error": { "type":"...", "message":"..." } }
	writeJSON(w, status, map[string]any{"type": "error", "error": map[string]any{"type": errorType, "message": message}})
}

func anthropicErrorTypeForStatus(status int) string {
	switch status {
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusServiceUnavailable, http.StatusGatewayTimeout, http.StatusBadGateway:
		return "api_error"
	default:
		return "api_error"
	}
}

// mapUpstreamStatusToAnthropic keeps client-facing status useful for Claude SDKs
// without inventing codes Anthropic clients reject.
func mapUpstreamStatusToAnthropic(status int) int {
	switch {
	case status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable:
		return status
	case status == http.StatusUnauthorized || status == http.StatusForbidden || status == http.StatusNotFound:
		return status
	case status == http.StatusBadRequest || status == http.StatusRequestEntityTooLarge:
		return status
	case status >= 500:
		return http.StatusBadGateway
	case status >= 400:
		return status
	default:
		return http.StatusBadGateway
	}
}

// unwrapAnthropicUpstreamMessage peels "upstream status N: body" wrappers and
// nested JSON error objects so tool-loop clients see the human/quota message.
func unwrapAnthropicUpstreamMessage(message string) (status int, body string, ok bool) {
	text := strings.TrimSpace(message)
	if text == "" {
		return 0, "", false
	}
	lower := strings.ToLower(text)
	for _, p := range []string{"upstream status ", "status "} {
		if !strings.HasPrefix(lower, p) {
			continue
		}
		rest := strings.TrimSpace(text[len(p):])
		i := 0
		for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
			status = status*10 + int(rest[i]-'0')
			i++
		}
		if status <= 0 || i == 0 {
			return 0, "", false
		}
		rest = strings.TrimSpace(rest[i:])
		if strings.HasPrefix(rest, ":") {
			rest = strings.TrimSpace(rest[1:])
		}
		return status, preferAnthropicErrorBody(rest), true
	}
	// Bare JSON body without status prefix.
	if text[0] == '{' {
		if cleaned := preferAnthropicErrorBody(text); cleaned != text {
			return 0, cleaned, true
		}
	}
	return 0, "", false
}

func preferAnthropicErrorBody(body string) string {
	body = strings.TrimSpace(body)
	if body == "" || body[0] != '{' {
		return body
	}
	var raw map[string]any
	if json.Unmarshal([]byte(body), &raw) != nil {
		return body
	}
	// {"error":"..."} or {"error":{"message":"..."}} or {"message":"..."}
	switch e := raw["error"].(type) {
	case string:
		if strings.TrimSpace(e) != "" {
			return strings.TrimSpace(e)
		}
	case map[string]any:
		if m, ok := e["message"].(string); ok && strings.TrimSpace(m) != "" {
			return strings.TrimSpace(m)
		}
		if m, ok := e["error"].(string); ok && strings.TrimSpace(m) != "" {
			return strings.TrimSpace(m)
		}
	}
	if m, ok := raw["message"].(string); ok && strings.TrimSpace(m) != "" {
		return strings.TrimSpace(m)
	}
	if m, ok := raw["detail"].(string); ok && strings.TrimSpace(m) != "" {
		return strings.TrimSpace(m)
	}
	return body
}

// anthropicErrorFromCause maps transport/upstream failures onto Anthropic
// terminal error type + human message for SSE event:error frames.
func anthropicErrorFromCause(err error) (message, errorType string) {
	if err == nil {
		return "request failed", "api_error"
	}
	message = strings.TrimSpace(err.Error())
	errorType = "api_error"
	var upstream *grok.UpstreamError
	if errors.As(err, &upstream) && upstream != nil {
		if strings.TrimSpace(upstream.Body) != "" {
			message = preferAnthropicErrorBody(upstream.Body)
		}
		errorType = anthropicErrorTypeForStatus(mapUpstreamStatusToAnthropic(upstream.Status))
		return firstNonEmptyStr(message, err.Error()), errorType
	}
	if status, body, ok := unwrapAnthropicUpstreamMessage(message); ok {
		if strings.TrimSpace(body) != "" {
			message = body
		}
		if status > 0 {
			errorType = anthropicErrorTypeForStatus(mapUpstreamStatusToAnthropic(status))
		}
	}
	// Empty upstream 200s are server-side transport issues for Claude clients.
	low := strings.ToLower(message)
	if strings.Contains(low, "empty model output") || strings.Contains(low, "no content/tool_calls") {
		errorType = "api_error"
	}
	if errors.Is(err, pool.ErrNoEligibleAccounts) {
		errorType = "api_error"
		if message == "" {
			message = "No eligible accounts available. All accounts may be in cooldown or disabled."
		}
	}
	return firstNonEmptyStr(message, "request failed"), errorType
}

func serveAdminStatus(w http.ResponseWriter, r *http.Request, options Options, protected bool) {
	if !options.AdminReadEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin read routes are not enabled"})
		return
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return
	}
	if protected {
		if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
			writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
			return
		}
	} else {
		// Unprotected /admin/api/status is polled every ~8s by the admin UI.
		// Serve a short-lived snapshot so reverse proxies never hit 502 on
		// 7k-account PoolSummary scans.
		statusPayloadMu.Lock()
		if statusPayloadCache != nil && time.Since(statusPayloadAt) < 3*time.Second {
			out := cloneStringAnyMap(statusPayloadCache)
			statusPayloadMu.Unlock()
			writeJSON(w, http.StatusOK, out)
			return
		}
		statusPayloadMu.Unlock()
	}
	store := options.Store
	accountCount, modelCount := int64(0), int64(0)
	keyStats := map[string]any{"total": int64(0), "enabled": int64(0), "disabled": int64(0), "total_requests": int64(0), "auth_required": false, "legacy_env_key": false}
	pool := postgres.PoolSummary{Mode: "round_robin", Source: "postgres"}
	if store != nil {
		if n, err := store.CountAccounts(r.Context()); err == nil {
			accountCount = n
		}
		if n, err := store.CountModels(r.Context(), false); err == nil {
			modelCount = n
		}
		if options.APIKeys != nil {
			if required, err := options.APIKeys.AuthRequired(r.Context()); err == nil {
				if stats, err := store.KeyStats(r.Context(), strings.TrimSpace(options.Config.LegacyAPIKey) != "", required); err == nil {
					keyStats = stats
				}
			}
		}
		if got, err := store.PoolSummary(r.Context()); err == nil {
			pool = got
		}
	}
	accounts := map[string]any{"account_count": accountCount, "active_count": pool.Live}
	setupNeeded := false
	if store != nil {
		if has, err := store.HasAdminPassword(r.Context()); err == nil {
			setupNeeded = !has
		}
	}
	// 构建前端兼容的数据库状态
	redisEnabled := options.Redis != nil && options.Redis.Enabled()
	redisConfigured := strings.TrimSpace(options.Config.RedisURL) != ""
	pgEnabled := store != nil
	pgConfigured := strings.TrimSpace(options.Config.DatabaseURL) != ""

	payload := map[string]any{
		"ok":           true,
		"setup_needed": setupNeeded,
		"version":      buildinfo.Version,
		"store": map[string]any{
			"backend": "hybrid",
			"postgres": map[string]any{
				"ok":         pgEnabled,
				"enabled":    pgEnabled,
				"configured": pgConfigured,
			},
			"redis": map[string]any{
				"ok":         redisEnabled,
				"enabled":    redisEnabled,
				"configured": redisConfigured,
			},
			"workers": options.Config.Workers,
		},
		"host":                 options.Config.Host,
		"port":                 options.Config.Port,
		"upstream":             options.Config.UpstreamBase,
		"default_model":        options.runtimeConfig().DefaultModel,
		"require_api_key_mode": options.Config.RequireAPIKey,
		"api_base":             publicAPIBase(r, options.Config.Port),
		"credentials_ok":       pool.Live > 0,
		"credentials_email":    nil,
		"account_mode":         pool.Mode,
		"accounts":             accounts,
		"pool": map[string]any{
			"mode": pool.Mode, "total": pool.Total, "live": pool.Live, "rotatable": pool.Rotatable,
			"enabled": pool.Enabled, "in_cooldown": pool.InCooldown, "quota_disabled": pool.QuotaDisabled,
			"model_blocked": pool.ModelBlocked, "expired": pool.Expired, "disabled": pool.Disabled, "source": pool.Source,
		},
		"keys":                  keyStats,
		"models_count":          modelCount,
		"settings":              map[string]any{},
		"token_maintainer":      serviceStatus(options.Maintainer, options),
		"model_health":          serviceStatus(options.ModelHealth, options),
		"conversation_affinity": map[string]any{"enabled": options.AffinityStore != nil, "implementation": "go"},
		"registration":          map[string]any{"mode": options.Config.RegistrationMode, "external": true, "available": options.Config.RegistrationServiceURL != ""},
		"usage":                 usageLightSnapshot(r.Context(), options),
		"leader":                leaderStatus(r.Context(), options),
		"redis":                 map[string]any{"enabled": redisEnabled, "prefix": options.Config.RedisPrefix},
		"stream":                streamSnapshot(),
	}
	if protected {
		payload["credentials"] = map[string]any{"email": nil, "active_count": pool.Live, "account_count": accountCount, "ok": pool.Live > 0}
		payload["models"] = modelCatalog(options).PublicModels(r.Context())
		payload["account_modes"] = []string{"round_robin", "random", "least_used"}
		payload["full"] = false
	} else {
		statusPayloadMu.Lock()
		statusPayloadCache = cloneStringAnyMap(payload)
		statusPayloadAt = time.Now()
		statusPayloadMu.Unlock()
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveAdminModels(w http.ResponseWriter, r *http.Request, options Options) {
	if !options.AdminReadEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin read routes are not enabled"})
		return
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object":        "list",
		"data":          modelCatalog(options).PublicModels(r.Context()),
		"default_model": options.runtimeConfig().DefaultModel,
		"storage":       "postgres",
		"meta":          map[string]any{},
	})
}

func serveAdminKeys(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	keys, err := options.Store.ListAPIKeys(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	public := make([]map[string]any, 0, len(keys))
	for _, key := range keys {
		public = append(public, key.PublicMap())
	}
	required := false
	if options.APIKeys != nil {
		required, _ = options.APIKeys.AuthRequired(r.Context())
	}
	stats, err := options.Store.KeyStats(r.Context(), strings.TrimSpace(options.Config.LegacyAPIKey) != "", required)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": public, "stats": stats, "store_source": "postgres", "store_backend": "postgres"})
}

func serveAdminAccounts(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	query := r.URL.Query()
	if truthy(query.Get("summary")) {
		count, _ := options.Store.CountAccounts(r.Context())
		pool, _ := options.Store.PoolSummary(r.Context())
		writeJSON(w, http.StatusOK, map[string]any{
			"account_count": count,
			"active_count":  pool.Live,
			"pool":          pool,
			"page":          1,
			"page_size":     0,
			"total":         count,
			"total_pages":   1,
			"q":             strings.TrimSpace(query.Get("q")),
			"sort":          query.Get("sort"),
		})
		return
	}
	page := intQuery(query.Get("page"), 1)
	pageSize := intQuery(query.Get("page_size"), 25)
	filter := postgres.AccountListFilter{
		Query:  query.Get("q"),
		Sort:   query.Get("sort"),
		Status: query.Get("status"),
	}
	if raw := strings.TrimSpace(strings.ToLower(query.Get("has_sso"))); raw != "" {
		v := raw == "1" || raw == "true" || raw == "yes" || raw == "on"
		// distinguish false explicitly
		if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
			v = false
			filter.HasSSO = &v
		} else if raw == "1" || raw == "true" || raw == "yes" || raw == "on" {
			v = true
			filter.HasSSO = &v
		}
	}
	if truthy(query.Get("ids_only")) || truthy(query.Get("ids")) {
		filter.IDsOnly = true
		// return up to 20k matching ids for "筛选全选"
		if pageSize < 1000 {
			pageSize = 20000
		}
		page = 1
	}
	result, err := options.Store.ListAccountSummariesFiltered(r.Context(), page, pageSize, filter)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	// Ensure frontend can re-sync filter chips.
	if result.Status == "" {
		result.Status = strings.TrimSpace(query.Get("status"))
	}
	// Always include pool as a plain map so chips/overview match DB even if
	// AccountList.Pool pointer encoding is stripped by proxies/older clients.
	payload := map[string]any{
		"accounts":       result.Accounts,
		"total":          result.Total,
		"page":           result.Page,
		"page_size":      result.PageSize,
		"total_pages":    result.TotalPages,
		"q":              result.Query,
		"sort":           result.Sort,
		"status":         result.Status,
		"store_source":   result.StoreSource,
		"store_backend":  result.StoreBackend,
		"auth_file_role": result.AuthFileRole,
	}
	if result.HasSSO != nil {
		payload["has_sso"] = *result.HasSSO
	}
	if len(result.IDs) > 0 {
		payload["ids"] = result.IDs
	}
	poolSum := result.Pool
	if poolSum == nil {
		if got, err := options.Store.PoolSummary(r.Context()); err == nil {
			poolSum = &got
		}
	}
	if poolSum != nil {
		payload["pool"] = map[string]any{
			"mode":           poolSum.Mode,
			"total":          poolSum.Total,
			"live":           poolSum.Live,
			"rotatable":      poolSum.Rotatable,
			"enabled":        poolSum.Enabled,
			"in_cooldown":    poolSum.InCooldown,
			"quota_disabled": poolSum.QuotaDisabled,
			"model_blocked":  poolSum.ModelBlocked,
			"expired":        poolSum.Expired,
			"disabled":       poolSum.Disabled,
			"source":         poolSum.Source,
		}
	}
	writeJSON(w, http.StatusOK, payload)
}

func registrationClient(options Options) *regclient.Client {
	base := strings.TrimSpace(options.RegistrationURL)
	if base == "" {
		base = strings.TrimSpace(options.Config.RegistrationServiceURL)
	}
	token := strings.TrimSpace(options.RegistrationToken)
	if token == "" {
		token = strings.TrimSpace(options.Config.RegistrationToken)
	}
	if base == "" {
		return nil
	}
	return &regclient.Client{BaseURL: base, Token: token}
}

func requireAdminReadWrite(w http.ResponseWriter, r *http.Request, options Options, write bool) bool {
	if write {
		if !adminWriteAllowed(w, r, options) {
			return false
		}
	} else if !adminRouteAllowed(w, r, options) {
		return false
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return false
	}
	return true
}

func recordRedisUsage(options Options, apiKeyID, accountID, model string, prompt, completion, total, cacheRead int64, ok bool) {
	if options.Redis == nil || !options.Redis.Enabled() {
		return
	}
	// Store billed tokens so /admin/api/status light snapshot does not over-count
	// prompt cache hits (and does not need a full PG UsageSummary scan).
	billed := total - cacheRead
	if billed < 0 {
		billed = 0
	}
	promptBilled := prompt - cacheRead
	if promptBilled < 0 {
		promptBilled = 0
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = options.Redis.RecordUsage(ctx, redis.UsageDeltas{
		PromptTokens:     promptBilled,
		CompletionTokens: completion,
		TotalTokens:      billed,
		OK:               ok,
		APIKeyID:         apiKeyID,
		AccountID:        accountID,
		Model:            model,
		TS:               time.Now().UTC(),
	})
}

func touchRedisPool(options Options, accountID string, success bool, errText string, cooldown *time.Time, status int) {
	if options.Redis == nil || !options.Redis.Enabled() || strings.TrimSpace(accountID) == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	touch := redis.PoolStatsTouch{Success: success, Error: errText, CooldownUntil: cooldown, LastStatusCode: &status}
	_, _ = options.Redis.TouchStats(ctx, accountID, touch)
}

// usageLight cache: /admin/api/status is polled every ~few seconds by the admin UI.
// Never run full UsageSummary / full-table scans here — reverse proxies
// (Cloudflare/nginx) return HTML 502 when this path stalls.
var (
	statusPayloadMu    sync.Mutex
	statusPayloadCache map[string]any
	statusPayloadAt    time.Time

	usageLightMu         sync.Mutex
	usageLightCache      map[string]any
	usageLightAt         time.Time
	usageLightPGAt       time.Time
	usageLightPGInFlight bool
)

func usageLightSnapshot(ctx context.Context, options Options) map[string]any {
	// 1) In-process cache — status auto-refresh must be free after first hit.
	usageLightMu.Lock()
	if usageLightCache != nil && time.Since(usageLightAt) < 20*time.Second {
		out := cloneStringAnyMap(usageLightCache)
		usageLightMu.Unlock()
		return out
	}
	usageLightMu.Unlock()

	// 2) Redis hot buckets (O(1)). Billed tokens are written at record time.
	if options.Redis != nil && options.Redis.Enabled() {
		snap := options.Redis.LightSnapshot(ctx)
		if snap != nil {
			// Normalize keys for overview card.
			if _, ok := snap["today_tokens"]; !ok {
				snap["today_tokens"] = snap["today_tokens"]
			}
			snap["source"] = "redis"
			usageLightMu.Lock()
			usageLightCache = cloneStringAnyMap(snap)
			usageLightAt = time.Now()
			// Rare PG reconcile (at most every 60s) — never on request path.
			needPG := time.Since(usageLightPGAt) > 60*time.Second && !usageLightPGInFlight
			if needPG {
				usageLightPGInFlight = true
			}
			usageLightMu.Unlock()
			if needPG {
				go refreshUsageLightFromPG(options)
			}
			return snap
		}
	}

	// 3) Stale cache better than a blocking PG scan under a reverse proxy.
	usageLightMu.Lock()
	if usageLightCache != nil {
		out := cloneStringAnyMap(usageLightCache)
		usageLightMu.Unlock()
		return out
	}
	usageLightMu.Unlock()

	// 4) Last resort: tiny PG light with hard 150ms budget (indexes on created_at).
	if options.Store != nil {
		cctx, cancel := context.WithTimeout(ctx, 150*time.Millisecond)
		light, err := options.Store.UsageLightBilled(cctx)
		cancel()
		if err == nil && light != nil {
			light["source"] = "postgres"
			usageLightMu.Lock()
			usageLightCache = cloneStringAnyMap(light)
			usageLightAt = time.Now()
			usageLightPGAt = time.Now()
			usageLightMu.Unlock()
			return light
		}
	}
	return map[string]any{"today_requests": 0, "today_tokens": 0, "total_tokens": 0, "source": "none"}
}

func refreshUsageLightFromPG(options Options) {
	defer func() {
		usageLightMu.Lock()
		usageLightPGInFlight = false
		usageLightMu.Unlock()
	}()
	if options.Store == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	light, err := options.Store.UsageLightBilled(ctx)
	if err != nil || light == nil {
		return
	}
	light["source"] = "postgres"
	usageLightMu.Lock()
	// Only replace redis snapshot if PG succeeded; keep 20s cache TTL from now.
	usageLightCache = cloneStringAnyMap(light)
	usageLightAt = time.Now()
	usageLightPGAt = time.Now()
	usageLightMu.Unlock()
}

func cloneStringAnyMap(in map[string]any) map[string]any {
	if in == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func leaderStatus(ctx context.Context, options Options) map[string]any {
	if options.Leader != nil {
		return options.Leader.Status(ctx)
	}
	return map[string]any{"is_leader": false, "mode": options.Config.MaintainerLeader, "implementation": "go", "started": false}
}

func maintainerStatus(options Options) map[string]any {
	started := options.Config.GoMaintainer && options.Leader != nil && options.Leader.IsLeader()
	return map[string]any{
		"enabled":         options.Config.GoMaintainer,
		"implementation":  "go",
		"started":         started,
		"leader_required": options.Config.Workers > 1,
	}
}

func serveRegistrationConfigGet(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	cfg, source := loadRegistrationConfig(r.Context(), options, true)
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":     true,
		"config": cfg,
		"source": source,
	})
}

func serveRegistrationConfigPut(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var body map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	cfg, err := saveRegistrationConfig(r.Context(), options, body, false)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":      true,
		"config":  cfg,
		"message": "注册配置已保存到数据库",
	})
}

func serveRegistrationProxyTest(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	if body == nil {
		body = map[string]any{}
	}
	// Prefer explicit body; fill blanks from saved registration_config.
	merged := mergeRegistrationStartBody(r.Context(), options, body)
	proxy := strings.TrimSpace(stringValue(merged["proxy"]))
	user := strings.TrimSpace(stringValue(merged["proxy_username"]))
	pass := strings.TrimSpace(stringValue(merged["proxy_password"]))
	strategy := strings.TrimSpace(stringValue(merged["proxy_strategy"]))
	if strategy == "" {
		strategy = "round_robin"
	}
	lines := splitProxyLines(proxy)
	summary := map[string]any{
		"enabled":  len(lines) > 0,
		"count":    len(lines),
		"strategy": strategy,
		"preview":  previewProxyLines(lines, 3),
		"source":   "registration_config",
	}
	if len(lines) == 0 {
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":            true,
			"proxy_enabled": false,
			"proxy_pool":    summary,
			"message":       "未配置代理（将直连）",
		})
		return
	}
	testAll := truthy(fmt.Sprint(body["test_all"]))
	maxTest := 5
	if v, ok := asInt(body["max_test"]); ok && v > 0 {
		maxTest = v
	}
	if maxTest > 20 {
		maxTest = 20
	}
	targets := lines
	if !testAll || len(lines) == 1 {
		targets = lines[:1]
	} else if len(targets) > maxTest {
		targets = targets[:maxTest]
	}
	results := make([]map[string]any, 0, len(targets))
	okN := 0
	for _, url := range targets {
		// Lightweight TCP/HTTP reachability is not captcha; report configured only.
		// Full proxy smoke stays in registration workers against xAI.
		item := map[string]any{
			"ok":      true,
			"proxy":   redactProxyURL(url, user),
			"message": "proxy entry accepted",
		}
		if user != "" && !strings.Contains(url, "@") {
			item["proxy_username"] = user
			item["has_password"] = pass != ""
		}
		results = append(results, item)
		okN++
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":            okN > 0,
		"proxy_enabled": true,
		"proxy_pool":    summary,
		"tested":        len(results),
		"ok_count":      okN,
		"fail_count":    len(results) - okN,
		"results":       results,
		"message":       fmt.Sprintf("validated %d/%d proxy entries", okN, len(lines)),
	})
}

var registrationSecretKeys = map[string]struct{}{
	"api_key":         {},
	"moemail_api_key": {},
	"yyds_api_key":    {},
	"gptmail_api_key": {},
	"cfmail_api_key":  {},
	"yescaptcha_key":  {},
	"proxy_password":  {},
}

func loadRegistrationConfig(ctx context.Context, options Options, includeSecrets bool) (map[string]any, string) {
	cfg := map[string]any{}
	source := "env"
	if options.Store != nil {
		if raw, err := options.Store.GetSetting(ctx, "registration_config"); err == nil {
			if m, ok := raw.(map[string]any); ok && m != nil {
				cfg = cloneMapAny(m)
				source = "database"
			}
		}
	}
	cfg = normalizeRegistrationConfig(cfg)
	if !includeSecrets {
		for k := range registrationSecretKeys {
			if v := strings.TrimSpace(stringValue(cfg[k])); v != "" {
				cfg[k] = maskSecret(v)
				cfg[k+"_set"] = true
			} else {
				cfg[k] = ""
				cfg[k+"_set"] = false
			}
		}
	}
	return cfg, source
}

func saveRegistrationConfig(ctx context.Context, options Options, patch map[string]any, replace bool) (map[string]any, error) {
	if options.Store == nil {
		return nil, errors.New("store unavailable")
	}
	if patch == nil {
		patch = map[string]any{}
	}
	current := map[string]any{}
	if !replace {
		if raw, err := options.Store.GetSetting(ctx, "registration_config"); err == nil {
			if m, ok := raw.(map[string]any); ok && m != nil {
				current = cloneMapAny(m)
			}
		}
	}
	// Merge patch: empty secrets keep previous; masked secrets keep previous.
	for k, v := range patch {
		if _, isSecret := registrationSecretKeys[k]; isSecret {
			s := strings.TrimSpace(fmt.Sprint(v))
			if s == "" || isMaskedSecret(s) {
				continue
			}
			current[k] = s
			continue
		}
		// Keep explicit empty strings for non-secrets (clears domain etc.).
		current[k] = v
	}
	current = normalizeRegistrationConfig(current)
	if err := options.Store.SetSetting(ctx, "registration_config", current); err != nil {
		return nil, err
	}
	return current, nil
}

func mergeRegistrationStartBody(ctx context.Context, options Options, body map[string]any) map[string]any {
	out := map[string]any{}
	saved, _ := loadRegistrationConfig(ctx, options, true)
	for k, v := range saved {
		out[k] = v
	}
	// Request overrides win when non-empty (secrets) or present (numbers/bools).
	for k, v := range body {
		if v == nil {
			continue
		}
		if _, isSecret := registrationSecretKeys[k]; isSecret {
			s := strings.TrimSpace(fmt.Sprint(v))
			if s == "" || isMaskedSecret(s) {
				continue
			}
			out[k] = s
			continue
		}
		switch vv := v.(type) {
		case string:
			if strings.TrimSpace(vv) == "" {
				// keep saved
				continue
			}
			out[k] = vv
		default:
			out[k] = v
		}
	}
	// Map form aliases used by UI/Python adapter.
	if apiKey := strings.TrimSpace(stringValue(out["api_key"])); apiKey != "" {
		provider := strings.ToLower(strings.TrimSpace(stringValue(out["mail_provider"])))
		switch provider {
		case "yyds":
			if strings.TrimSpace(stringValue(out["yyds_api_key"])) == "" {
				out["yyds_api_key"] = apiKey
			}
		case "gptmail":
			if strings.TrimSpace(stringValue(out["gptmail_api_key"])) == "" {
				out["gptmail_api_key"] = apiKey
			}
		case "cfmail":
			if strings.TrimSpace(stringValue(out["cfmail_api_key"])) == "" {
				out["cfmail_api_key"] = apiKey
			}
		default:
			if strings.TrimSpace(stringValue(out["moemail_api_key"])) == "" {
				out["moemail_api_key"] = apiKey
			}
		}
	}
	// Adapter expects moemail_* names for mail credentials historically.
	if strings.TrimSpace(stringValue(out["moemail_api_key"])) == "" {
		if v := strings.TrimSpace(stringValue(out["api_key"])); v != "" {
			out["moemail_api_key"] = v
		}
	}
	if strings.TrimSpace(stringValue(out["moemail_base_url"])) == "" {
		if v := strings.TrimSpace(stringValue(out["base_url"])); v != "" {
			out["moemail_base_url"] = v
		}
	}
	return out
}

func normalizeRegistrationConfig(raw map[string]any) map[string]any {
	cfg := map[string]any{}
	if raw != nil {
		for k, v := range raw {
			cfg[k] = v
		}
	}
	// defaults
	if strings.TrimSpace(stringValue(cfg["mail_provider"])) == "" {
		cfg["mail_provider"] = "moemail"
	}
	provider := strings.ToLower(strings.TrimSpace(stringValue(cfg["captcha_provider"])))
	if provider != "yescaptcha" {
		provider = "local"
	}
	cfg["captcha_provider"] = provider
	if provider == "local" {
		cfg["local_solver_url"] = "http://127.0.0.1:5072"
	}
	strat := strings.ToLower(strings.TrimSpace(stringValue(cfg["proxy_strategy"])))
	switch strat {
	case "random", "rand":
		cfg["proxy_strategy"] = "random"
	case "sticky", "first", "fixed":
		cfg["proxy_strategy"] = "sticky"
	default:
		cfg["proxy_strategy"] = "round_robin"
	}
	// numeric defaults
	if _, ok := asInt(cfg["count"]); !ok {
		cfg["count"] = 1
	}
	if _, ok := asInt(cfg["concurrency"]); !ok {
		cfg["concurrency"] = 3
	}
	if _, ok := asInt(cfg["stagger_ms"]); !ok {
		cfg["stagger_ms"] = 300
	}
	if _, ok := asInt(cfg["probe_delay_sec"]); !ok {
		cfg["probe_delay_sec"] = 30
	}
	return cfg
}

func maskSecret(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return ""
	}
	if len(v) <= 4 {
		return "****"
	}
	return v[:2] + "…" + v[len(v)-2:]
}

func isMaskedSecret(v string) bool {
	s := strings.TrimSpace(v)
	if s == "" {
		return false
	}
	if s == "****" {
		return true
	}
	if strings.Contains(s, "…") || strings.Contains(s, "...") {
		return true
	}
	for _, ch := range s {
		if ch != '*' {
			return false
		}
	}
	return true
}

func splitProxyLines(text string) []string {
	raw := strings.FieldsFunc(text, func(r rune) bool {
		return r == '\n' || r == '\r' || r == ';' || r == ','
	})
	out := make([]string, 0, len(raw))
	for _, line := range raw {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out = append(out, line)
	}
	return out
}

func previewProxyLines(lines []string, n int) []string {
	if n <= 0 || len(lines) == 0 {
		return []string{}
	}
	if len(lines) < n {
		n = len(lines)
	}
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, redactProxyURL(lines[i], ""))
	}
	return out
}

func redactProxyURL(raw, user string) string {
	s := strings.TrimSpace(raw)
	if s == "" {
		return ""
	}
	// hide password in user:pass@host
	if at := strings.LastIndex(s, "@"); at > 0 {
		head := s[:at]
		host := s[at+1:]
		if i := strings.Index(head, "://"); i >= 0 {
			scheme := head[:i+3]
			cred := head[i+3:]
			if colon := strings.Index(cred, ":"); colon >= 0 {
				return scheme + cred[:colon] + ":***@" + host
			}
		}
	}
	if user != "" && !strings.Contains(s, "@") {
		return s + " (auth user set)"
	}
	return s
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	case string:
		s := strings.TrimSpace(n)
		if s == "" {
			return 0, false
		}
		i, err := strconv.Atoi(s)
		return i, err == nil
	default:
		return 0, false
	}
}

func cloneMapAny(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func writeRegistrationError(w http.ResponseWriter, err error) {
	var re *regclient.Error
	if errors.As(err, &re) {
		status := re.Status
		if status < 400 {
			status = http.StatusBadGateway
		}
		writeJSON(w, status, map[string]any{"detail": re.Detail})
		return
	}
	writeJSON(w, http.StatusBadGateway, map[string]any{"detail": err.Error()})
}

func serveDeviceLoginStart(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration/device service URL is not configured"})
		return
	}
	var body map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&body); err != nil && !errors.Is(err, io.EOF) {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	if body == nil {
		body = map[string]any{"mode": "device", "capture": true}
	}
	if _, ok := body["mode"]; !ok {
		body["mode"] = "device"
	}
	payload, err := client.StartDeviceLogin(r.Context(), body)
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveDeviceLoginSession(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration/device service URL is not configured"})
		return
	}
	sid := strings.TrimSpace(r.PathValue("session_id"))
	if sid == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "session_id required"})
		return
	}
	payload, err := client.DeviceLoginSession(r.Context(), sid)
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveDeviceLoginSessions(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration/device service URL is not configured"})
		return
	}
	payload, err := client.DeviceLoginSessions(r.Context())
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveSSOImportStart(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration/sso service URL is not configured"})
		return
	}
	var body map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	payload, err := client.StartSSOImport(r.Context(), body)
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveSSOImportJob(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration/sso service URL is not configured"})
		return
	}
	payload, err := client.SSOImportJob(r.Context(), r.PathValue("job_id"))
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationAvailability(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured", "ok": false, "available": false})
		return
	}
	payload, err := client.Availability(r.Context())
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationSessions(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	payload, err := client.Sessions(r.Context())
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationSession(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	includeAuth := truthy(r.URL.Query().Get("include_auth_json"))
	payload, err := client.Session(r.Context(), r.PathValue("session_id"), includeAuth)
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationStopSession(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	payload, err := client.StopSession(r.Context(), r.PathValue("session_id"))
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationBatch(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	payload, err := client.Batch(r.Context(), r.PathValue("batch_id"))
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationStopBatch(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	payload, err := client.StopBatch(r.Context(), r.PathValue("batch_id"))
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationResumeBatch(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	force, _ := body["force"].(bool)
	payload, err := client.ResumeBatch(r.Context(), r.PathValue("batch_id"), force)
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationStart(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	var body map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&body); err != nil && !errors.Is(err, io.EOF) {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	if body == nil {
		body = map[string]any{}
	}
	// Merge non-empty request overrides onto durable registration_config so
	// empty form fields still use last-saved mail/captcha/proxy secrets.
	body = mergeRegistrationStartBody(r.Context(), options, body)
	// Auto-persist last-used config (secrets only when newly provided).
	if options.Store != nil {
		if _, err := saveRegistrationConfig(r.Context(), options, body, false); err != nil {
			// non-fatal: registration can still start with merged body
		}
	}
	idem := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	if idem == "" {
		idem = strings.TrimSpace(stringValue(body["idempotency_key"]))
	}
	payload, err := client.Start(r.Context(), body, idem)
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationReclaim(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	autoResume := true
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err == nil {
		if v, ok := body["auto_resume"].(bool); ok {
			autoResume = v
		}
	}
	payload, err := client.Reclaim(r.Context(), autoResume)
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveRegistrationStopAll(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	client := registrationClient(options)
	if client == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "registration service URL is not configured"})
		return
	}
	payload, err := client.StopAll(r.Context())
	if err != nil {
		writeRegistrationError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveAdminUpdateSettings(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	var patch map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&patch); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	settings, err := options.Store.UpdateRuntimeSettings(r.Context(), patch)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	options.applySettingsToRuntime(settings)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "settings": settings})
}

func serveAdminSettings(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	settings, err := options.Store.PublicSettings(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "settings": settings})
}

func serveAdminLogs(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	query := r.URL.Query()
	items, err := options.Store.ListTasks(r.Context(), intQuery(query.Get("page"), 1), intQuery(query.Get("page_size"), 50), query.Get("q"), firstNonEmpty(query.Get("kind"), query.Get("action")), query.Get("status"))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, items)
}

func serveAdminLogActions(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	kinds, err := options.Store.ListTaskKinds(r.Context(), 50)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error(), "actions": kinds})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "actions": kinds, "kinds": kinds})
}

func serveUsageSummary(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	payload, err := options.Store.UsageSummary(r.Context(), intQuery(r.URL.Query().Get("days"), 7))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveUsageSeries(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	payload, err := options.Store.UsageSeries(r.Context(), intQuery(r.URL.Query().Get("days"), 7))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveUsageBreakdown(w http.ResponseWriter, r *http.Request, options Options, dim string) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	query := r.URL.Query()
	payload, err := options.Store.UsageBreakdown(r.Context(), dim, intQuery(query.Get("days"), 7), intQuery(query.Get("limit"), 50))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func serveUsageEvents(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminRouteAllowed(w, r, options) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	query := r.URL.Query()
	var okFlag *bool
	switch strings.ToLower(strings.TrimSpace(query.Get("ok"))) {
	case "1", "true", "yes", "ok", "success":
		v := true
		okFlag = &v
	case "0", "false", "no", "fail", "failed", "error":
		v := false
		okFlag = &v
	}
	payload, err := options.Store.UsageEvents(r.Context(), intQuery(query.Get("page"), 1), intQuery(query.Get("page_size"), 50), map[string]string{
		"q": query.Get("q"), "api_key_id": query.Get("api_key_id"), "account_id": query.Get("account_id"), "model": query.Get("model"), "protocol": query.Get("protocol"), "client_ip": query.Get("client_ip"),
	}, okFlag)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, payload)
}

func adminRouteAllowed(w http.ResponseWriter, r *http.Request, options Options) bool {
	if !options.AdminReadEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin read routes are not enabled"})
		return false
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return false
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return false
	}
	return true
}

func truthy(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func intQuery(value string, fallback int) int {
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil {
		return fallback
	}
	return parsed
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func upstreamClient(options Options) *grok.Client {
	if options.Upstream != nil {
		return options.Upstream
	}
	return &grok.Client{BaseURL: options.Config.UpstreamBase}
}

// resolvePickMode returns the configured account rotation mode for failover chains.
// Sticky accounts are still forced first; mode only ranks the rest of the window.
func resolvePickMode(options Options) string {
	mode := "least_used"
	if options.Store != nil {
		if raw, err := options.Store.GetSetting(context.Background(), "account_mode"); err == nil {
			if s, ok := raw.(string); ok {
				if v := strings.ToLower(strings.TrimSpace(s)); v != "" {
					mode = v
				}
			}
		}
	}
	switch mode {
	case "round_robin", "random", "least_used":
		return mode
	default:
		return "least_used"
	}
}

func listCandidatesForRequest(ctx context.Context, options Options, chatReq proxy.ChatRequest, headers http.Header) ([]pool.Candidate, error) {
	candidates, err := listCandidates(ctx, options)
	if err != nil {
		return nil, err
	}
	fp := proxy.ChatFingerprint(chatReq)
	if fp == "" {
		fp = proxy.ChatFingerprintFromHeaders(headers, chatReq.Model)
	}
	// If still empty, try raw prompt_cache_key alone.
	if fp == "" && chatReq.Raw != nil {
		if pck, _ := chatReq.Raw["prompt_cache_key"].(string); strings.TrimSpace(pck) != "" {
			fp = "chat:" + strings.TrimSpace(chatReq.Model) + ":prompt_cache_key:" + strings.TrimSpace(pck)
		}
	}
	sticky := stickyAccountID(ctx, options, fp)
	// Direct pck recovery when fingerprint shape differed or model was empty.
	if sticky == "" && chatReq.Raw != nil && options.AffinityStore != nil {
		if pck, _ := chatReq.Raw["prompt_cache_key"].(string); strings.TrimSpace(pck) != "" {
			pck = strings.TrimSpace(pck)
			if id, err := options.AffinityStore.GetAffinity(ctx, "chat:prompt_cache_key:"+pck); err == nil {
				sticky = strings.TrimSpace(id)
			}
			if sticky == "" && strings.TrimSpace(chatReq.Model) != "" {
				if id, err := options.AffinityStore.GetAffinity(ctx, "chat:"+strings.TrimSpace(chatReq.Model)+":prompt_cache_key:"+pck); err == nil {
					sticky = strings.TrimSpace(id)
				}
			}
		}
	}
	// Codex previous_response_id recovery when fingerprint miss.
	if sticky == "" && chatReq.Raw != nil {
		if prev, _ := chatReq.Raw["previous_response_id"].(string); strings.TrimSpace(prev) != "" {
			if acc, recoveredPCK := responseAffinityLookup(ctx, options, prev); acc != "" {
				sticky = acc
				// If pck was recovered, also ensure request carries it for upstream cache.
				if recoveredPCK != "" {
					if _, has := chatReq.Raw["prompt_cache_key"]; !has || strings.TrimSpace(stringValue(chatReq.Raw["prompt_cache_key"])) == "" {
						chatReq.Raw["prompt_cache_key"] = recoveredPCK
					}
				}
			}
		}
	}
	candidates = ensureStickyCandidate(ctx, options, candidates, sticky)
	// Pin sticky to front so prepareChain can skip a second Redis affinity GET.
	if sticky != "" {
		for i := range candidates {
			if candidates[i].ID == sticky {
				if i > 0 {
					cand := candidates[i]
					copy(candidates[1:i+1], candidates[0:i])
					candidates[0] = cand
				}
				// Massive least_used boost so Chain keeps sticky first.
				candidates[0].RequestCount -= 1_000_000_000
				break
			}
		}
	}
	return candidates, nil
}

func listCandidates(ctx context.Context, options Options) ([]pool.Candidate, error) {
	if len(options.Candidates) > 0 {
		out := make([]pool.Candidate, len(options.Candidates))
		copy(out, options.Candidates)
		return out, nil
	}
	if options.Store == nil {
		return nil, errors.New("PostgreSQL store unavailable")
	}
	return options.Store.ListPoolCandidates(ctx)
}

// ensureStickyCandidate injects an affinity-bound account into the pick window
// even when it is outside the top-N least_used scan. Critical for prompt-cache
// multi-turn (Codex prompt_cache_key / X-Grok-Conv-Id) high hit rates.
func ensureStickyCandidate(ctx context.Context, options Options, candidates []pool.Candidate, stickyID string) []pool.Candidate {
	stickyID = strings.TrimSpace(stickyID)
	if stickyID == "" || options.Store == nil {
		return candidates
	}
	for _, c := range candidates {
		if c.ID == stickyID {
			return candidates
		}
	}
	extra, err := options.Store.GetPoolCandidate(ctx, stickyID)
	if err != nil || extra == nil {
		return candidates
	}
	// Skip cooled-down / disabled / expired sticky pins — injecting them only
	// burns TTFT and forces a known-bad first hop every multi-turn request.
	if !extra.Eligible("", time.Now()) {
		return candidates
	}
	// Put sticky first so prepareChain affinity boost is redundant but safe.
	out := make([]pool.Candidate, 0, len(candidates)+1)
	out = append(out, *extra)
	out = append(out, candidates...)
	return out
}

func stickyAccountID(ctx context.Context, options Options, fingerprint string) string {
	fingerprint = strings.TrimSpace(fingerprint)
	if fingerprint == "" || options.AffinityStore == nil {
		return ""
	}
	id, err := options.AffinityStore.GetAffinity(ctx, fingerprint)
	if err == nil {
		if s := strings.TrimSpace(id); s != "" {
			return s
		}
	}
	// Fallbacks for fingerprint shape drift across turns / model aliasing.
	// chat:{model}:prompt_cache_key:{pck} -> chat:prompt_cache_key:{pck}
	if i := strings.Index(fingerprint, ":prompt_cache_key:"); i >= 0 {
		alt := "chat" + fingerprint[i:]
		if alt != fingerprint {
			if id, err := options.AffinityStore.GetAffinity(ctx, alt); err == nil {
				if s := strings.TrimSpace(id); s != "" {
					return s
				}
			}
		}
	}
	// chat:{model}:previous_response_id:{id} -> chat:previous_response_id:{id}
	if i := strings.Index(fingerprint, ":previous_response_id:"); i >= 0 {
		alt := "chat" + fingerprint[i:]
		if alt != fingerprint {
			if id, err := options.AffinityStore.GetAffinity(ctx, alt); err == nil {
				if s := strings.TrimSpace(id); s != "" {
					return s
				}
			}
		}
	}
	return ""
}

func modelCatalog(options Options) *models.Catalog {
	if options.Models != nil {
		return options.Models
	}
	return models.NewCatalog(config.Config{DefaultModel: "grok-4.5"}, nil)
}

func publicAPIBase(r *http.Request, port int) string {
	host := r.Host
	if forwarded := strings.TrimSpace(r.Header.Get("X-Forwarded-Host")); forwarded != "" {
		host = forwarded
	}
	proto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto"))
	if proto == "" {
		proto = "http"
	}
	if strings.TrimSpace(host) == "" {
		host = "127.0.0.1"
		if port > 0 {
			host = host + ":" + itoaPort(port)
		}
	}
	return proto + "://" + host + "/v1"
}

func serveAdminSetAccountEnabled(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	enabled, ok := body["enabled"].(bool)
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "enabled bool required"})
		return
	}
	rec, err := options.Store.SetAccountEnabled(r.Context(), r.PathValue("account_id"), enabled)
	if err != nil {
		if postgres.IsAccountNotFound(err) {
			writeJSON(w, http.StatusNotFound, map[string]any{"detail": "Account not found"})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "account": rec})
}

func serveAdminImportAccount(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var body map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	merge := true
	if v, ok := body["merge"].(bool); ok {
		merge = v
	}
	payload := body["payload"]
	if payload == nil {
		// allow bare auth object / token fields at top level
		payload = body
	}
	normalized := accounts.CollectNormalizedEntries(payload)
	if !normalized.OK {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "detail": normalized.Error, "error": normalized.Error})
		return
	}
	result, err := options.Store.ImportNormalizedAccounts(r.Context(), normalized.Normalized, merge)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if normalized.Format != "" {
		result["format"] = normalized.Format
	}
	writeJSON(w, http.StatusOK, result)
}

func serveAdminExportAccounts(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	includeSecrets := r.URL.Query().Get("include_secrets") != "0" && r.URL.Query().Get("include_secrets") != "false"
	asyncJob := truthy(r.URL.Query().Get("async_job"))
	result, err := options.Store.ExportAuthMap(r.Context(), nil, includeSecrets)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if asyncJob {
		writeJSON(w, http.StatusOK, startExportJob(result, "grok2api-auth-export.json"))
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func serveAdminExportAccountsBatch(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	includeSecrets := true
	if v, ok := body["include_secrets"].(bool); ok {
		includeSecrets = v
	}
	ids := stringSlice(body["ids"])
	asyncJob := truthy(r.URL.Query().Get("async_job")) || truthy(fmt.Sprint(body["async_job"]))
	result, err := options.Store.ExportAuthMap(r.Context(), ids, includeSecrets)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if asyncJob {
		name := fmt.Sprintf("grok2api-auth-export-selected-%d.json", len(ids))
		writeJSON(w, http.StatusOK, startExportJob(result, name))
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func serveAdminDeleteAccount(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	accountID := r.PathValue("account_id")
	if strings.HasPrefix(accountID, "register-email") || strings.Contains(accountID, "/register-email") {
		writeJSON(w, http.StatusNotFound, map[string]any{"detail": "Not found"})
		return
	}
	ok, err := options.Store.DeleteAccount(r.Context(), accountID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]any{"detail": "Account not found"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func serveAdminDeleteAccountsBatch(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	ids := stringSlice(body["ids"])
	if len(ids) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "ids is required"})
		return
	}
	if len(ids) > 2000 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "too many ids (max 2000)"})
		return
	}
	result, err := options.Store.DeleteAccounts(r.Context(), ids)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	result["ok"] = true
	writeJSON(w, http.StatusOK, result)
}

func serveAdminClearAllAccounts(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	n, err := options.Store.ClearAllAccounts(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":      true,
		"message": "已清空账号池",
		"removed": n,
	})
}

func stringSlice(value any) []string {
	switch v := value.(type) {
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s := stringValue(item); s != "" {
				out = append(out, s)
			}
		}
		return out
	case []string:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s := strings.TrimSpace(item); s != "" {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func serveAdminProbeBatch(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.ModelHealth == nil || options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "model health unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	ids := stringSlice(body["ids"])
	if len(ids) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "ids is empty"})
		return
	}
	if len(ids) > 500 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "too many ids (max 500)"})
		return
	}
	model := stringValue(body["model"])
	autoDisable := true
	if v, ok := body["auto_disable"].(bool); ok {
		autoDisable = v
	}
	results := options.ModelHealth.ProbeIDs(r.Context(), ids, model, autoDisable, "manual")
	// attach pool views
	out := make([]map[string]any, 0, len(results))
	for _, item := range results {
		aid := stringValue(item["account_id"])
		if pool, err := options.Store.GetAccountPoolView(r.Context(), aid); err == nil {
			item["pool"] = pool
		}
		out = append(out, item)
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "results": out, "count": len(out)})
}

func serveAdminProbeAll(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.ModelHealth == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "model health unavailable"})
		return
	}
	// Default: async multi-wave job so large pools are not truncated by one
	// 150s budget / HTTP timeout. ?sync=1 forces a blocking single response
	// (still multi-wave until covered or job timeout).
	sync := r.URL.Query().Get("sync") == "1" || r.URL.Query().Get("sync") == "true"
	if sync {
		// Bound the request context so a hung upstream cannot pin the handler forever.
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Minute)
		defer cancel()
		result := options.ModelHealth.RunOnce(ctx, "manual_all")
		writeJSON(w, http.StatusOK, result)
		return
	}
	result := options.ModelHealth.StartProbeAll()
	writeJSON(w, http.StatusOK, result)
}

func serveModelHealthStatus(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.ModelHealth == nil {
		writeJSON(w, http.StatusOK, map[string]any{"enabled": false, "implementation": "go", "started": false})
		return
	}
	writeJSON(w, http.StatusOK, options.ModelHealth.Status())
}

func serveMaintainerStatus(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Maintainer == nil {
		writeJSON(w, http.StatusOK, map[string]any{"enabled": false, "implementation": "go", "started": false})
		return
	}
	writeJSON(w, http.StatusOK, options.Maintainer.Status())
}

func serveMaintainerRun(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Maintainer == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "maintainer unavailable"})
		return
	}
	force := true
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err == nil {
		if v, ok := body["force"].(bool); ok {
			force = v
		}
	}
	writeJSON(w, http.StatusOK, options.Maintainer.RunOnce(r.Context(), force))
}

func serveAccountsRefresh(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Maintainer == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "maintainer unavailable"})
		return
	}
	force := true
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	if v, ok := body["force"].(bool); ok {
		force = v
	}
	ids := stringSlice(body["ids"])
	if len(ids) == 0 {
		ids = stringSlice(body["account_ids"])
	}
	var result map[string]any
	if len(ids) > 0 {
		// Selected-account renew: refresh only those ids and return per-row results
		// so the admin UI can paint immediately (busy rows / toast / pool patch).
		result = options.Maintainer.RunForIDs(r.Context(), ids, force)
	} else {
		result = options.Maintainer.RunOnce(r.Context(), force)
	}
	result["maintainer"] = options.Maintainer.Status()
	result["token_maintainer"] = result["maintainer"]
	writeJSON(w, http.StatusOK, result)
}

func serveToggleTokenMaintain(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	enabled, ok := body["enabled"].(bool)
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "enabled bool required"})
		return
	}
	if err := options.Store.SetSetting(r.Context(), "token_maintain_enabled", enabled); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if options.Maintainer != nil {
		if enabled {
			options.Maintainer.Start()
			options.Maintainer.RequestRunSoon(false)
		} else {
			options.Maintainer.Stop()
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":                     true,
		"token_maintain_enabled": enabled,
		"settings":               map[string]any{"token_maintain_enabled": enabled},
		"maintainer":             serviceStatus(options.Maintainer, options),
		"token_maintainer":       serviceStatus(options.Maintainer, options),
	})
}

func serveToggleModelHealth(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	enabled, ok := body["enabled"].(bool)
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "enabled bool required"})
		return
	}
	if err := options.Store.SetSetting(r.Context(), "model_health_enabled", enabled); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if options.ModelHealth != nil {
		if enabled {
			options.ModelHealth.Start()
			options.ModelHealth.RequestRunSoon()
		} else {
			options.ModelHealth.Stop()
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":                   true,
		"model_health_enabled": enabled,
		"settings":             map[string]any{"model_health_enabled": enabled},
		"model_health":         serviceStatus(options.ModelHealth, options),
	})
}

func serveSetAccountMode(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	mode := strings.ToLower(stringValue(body["mode"]))
	switch mode {
	case "round_robin", "random", "least_used":
	default:
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "Invalid account_mode. Use one of: round_robin, random, least_used"})
		return
	}
	if err := options.Store.SetSetting(r.Context(), "account_mode", mode); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "account_mode": mode, "modes": []string{"round_robin", "random", "least_used"}})
}

func serveChangeAdminPassword(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	current := stringValue(body["current_password"])
	newPW := stringValue(body["new_password"])
	confirm := stringValue(body["confirm_password"])
	if len(newPW) < 4 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "password must contain at least 4 characters"})
		return
	}
	if confirm != "" && confirm != newPW {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "两次输入的新密码不一致"})
		return
	}
	ok, err := verifyAdminPassword(r.Context(), options, current)
	if err != nil || !ok {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "当前密码不正确"})
		return
	}
	hash, salt, err := adminauth.NewPassword(newPW)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if err := options.Store.SetAdminPassword(r.Context(), hash, salt); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	settings, _ := options.Store.PublicSettings(r.Context())
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "message": "密码已更新", "settings": settings})
}

func servePruneModelBlocks(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	n, err := options.Store.PruneModelBlocks(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "pruned": n})
}

func serveExportAccountsSSO(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	authMap, err := options.Store.ExportAuthMap(r.Context(), nil, true)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, buildSSOExport(authMap))
}

func serveExportAccountsSSOSelected(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	ids := stringSlice(body["ids"])
	authMap, err := options.Store.ExportAuthMap(r.Context(), ids, true)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, buildSSOExport(authMap))
}

func serveAdminImportFile(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		// also accept JSON body {payload, merge}
		var body map[string]any
		if jerr := json.NewDecoder(r.Body).Decode(&body); jerr != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
			return
		}
		merge := true
		if v, ok := body["merge"].(bool); ok {
			merge = v
		}
		norm := accounts.CollectNormalizedEntries(body["payload"])
		if !norm.OK {
			writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": norm.Error})
			return
		}
		result, err := options.Store.ImportNormalizedAccounts(r.Context(), norm.Normalized, merge)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, result)
		return
	}
	merge := true
	if v := r.FormValue("merge"); v == "0" || strings.EqualFold(v, "false") {
		merge = false
	}
	file, _, err := r.FormFile("file")
	if err != nil {
		// try "files"
		file, _, err = r.FormFile("files")
	}
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "file required"})
		return
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, 16<<20))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	norm := accounts.CollectNormalizedEntries(string(raw))
	if !norm.OK {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": norm.Error})
		return
	}
	result, err := options.Store.ImportNormalizedAccounts(r.Context(), norm.Normalized, merge)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if norm.Format != "" {
		result["format"] = norm.Format
	}
	writeJSON(w, http.StatusOK, result)
}

func serveAdminImportFiles(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	merge := true
	if v := r.FormValue("merge"); v == "0" || strings.EqualFold(v, "false") {
		merge = false
	}
	files := r.MultipartForm.File["files"]
	if len(files) == 0 {
		files = r.MultipartForm.File["file"]
	}
	if len(files) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "files required"})
		return
	}
	normalized := map[string]map[string]any{}
	fileResults := []map[string]any{}
	parseErrors := 0
	for i, fh := range files {
		f, err := fh.Open()
		if err != nil {
			parseErrors++
			fileResults = append(fileResults, map[string]any{"index": i + 1, "ok": false, "error": err.Error()})
			continue
		}
		raw, _ := io.ReadAll(io.LimitReader(f, 16<<20))
		f.Close()
		norm := accounts.CollectNormalizedEntries(string(raw))
		if !norm.OK {
			parseErrors++
			fileResults = append(fileResults, map[string]any{"index": i + 1, "ok": false, "error": norm.Error, "format": norm.Format})
			continue
		}
		for k, v := range norm.Normalized {
			normalized[k] = v
		}
		fileResults = append(fileResults, map[string]any{"index": i + 1, "ok": true, "count": len(norm.Normalized), "format": norm.Format})
	}
	if len(normalized) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "no valid account entries found", "file_results": fileResults, "parse_errors": parseErrors})
		return
	}
	result, err := options.Store.ImportNormalizedAccounts(r.Context(), normalized, merge)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	result["files"] = len(files)
	result["parse_errors"] = parseErrors
	result["file_results"] = fileResults
	writeJSON(w, http.StatusOK, result)
}

func serveAdminNormalizeAccounts(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	result, err := options.Store.NormalizeAccountKeys(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func serveAdminModelsSync(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	authList, err := options.Store.ListAccountAuths(r.Context(), 20, true)
	if err != nil || len(authList) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "no live account for models sync"})
		return
	}
	a := authList[0]
	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, strings.TrimRight(options.Config.UpstreamBase, "/")+"/models", nil)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	gc := upstreamClient(options)
	for k, v := range gc.Headers(a.Token, options.runtimeConfig().DefaultModel) {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode >= 400 {
		writeJSON(w, http.StatusBadGateway, map[string]any{"ok": false, "error": fmt.Sprintf("upstream %d: %s", resp.StatusCode, string(body)[:minInt(300, len(body))])})
		return
	}
	var payload any
	if err := json.Unmarshal(body, &payload); err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"ok": false, "error": "parse: " + err.Error()})
		return
	}
	items := parseUpstreamModels(payload)
	// Always keep local extras (coding/build/search aliases) even when upstream list is tiny.
	items = ensureLocalModelExtras(items, options.runtimeConfig().DefaultModel)
	if len(items) == 0 {
		writeJSON(w, http.StatusBadGateway, map[string]any{"ok": false, "error": "no models in upstream response"})
		return
	}
	n, err := options.Store.ReplaceModels(r.Context(), items, map[string]any{"source": "upstream", "fetched_via": a.Email, "origin": strings.TrimRight(options.Config.UpstreamBase, "/") + "/models"})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	catalog := modelCatalog(options).PublicModels(r.Context())
	writeJSON(w, http.StatusOK, map[string]any{
		"ok": true, "count": len(catalog), "pg_count": n, "upstream_count": n,
		"fetched_via": a.Email, "storage": "postgres", "models": catalog, "data": catalog,
	})
}

func serveAdminAccountsQuota(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Quota == nil {
		// fallback store-only cached
		if options.Store == nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
			return
		}
		out, err := options.Store.ListCachedQuotas(r.Context())
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, out)
		return
	}
	cached := r.URL.Query().Get("cached") == "1" || r.URL.Query().Get("cached") == "true"
	refresh := r.URL.Query().Get("refresh") == "1" || r.URL.Query().Get("refresh") == "true"
	if cached && !refresh {
		out, err := options.Quota.FetchCached(r.Context())
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, out)
		return
	}
	out, err := options.Quota.FetchAll(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func serveAdminAccountQuota(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	aid := strings.TrimSpace(r.PathValue("account_id"))
	if aid == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "account_id required"})
		return
	}
	// Live fetch by default so "额度" button always works even without cache.
	// ?cached=1 keeps old behavior for bulk UI hydration.
	cachedOnly := truthy(r.URL.Query().Get("cached")) && !truthy(r.URL.Query().Get("refresh"))
	if cachedOnly {
		all, err := options.Store.ListCachedQuotas(r.Context())
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
			return
		}
		results, _ := all["results"].([]map[string]any)
		if results == nil {
			if arr, ok := all["results"].([]any); ok {
				for _, item := range arr {
					if m, ok := item.(map[string]any); ok {
						results = append(results, m)
					}
				}
			}
		}
		for _, item := range results {
			if stringValue(item["account_id"]) == aid {
				writeJSON(w, http.StatusOK, item)
				return
			}
		}
		writeJSON(w, http.StatusNotFound, map[string]any{"detail": "quota cache not found"})
		return
	}
	if options.Quota == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "quota service unavailable"})
		return
	}
	item, err := options.Quota.FetchOne(r.Context(), aid)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if item == nil {
		item = map[string]any{"ok": false, "account_id": aid, "error": "empty quota result"}
	}
	// Prefer the synthetic pool from FetchOne (live billing result) so the admin
	// UI paints immediately. DB SaveQuotaSnapshot is already async; if a durable
	// pool view is available and not lagging, merge cooldown flags from it.
	if poolView, perr := options.Store.GetAccountPoolView(r.Context(), aid); perr == nil && poolView != nil {
		// Keep synthetic last_quota / quota_disabled from FetchOne; only enrich
		// non-quota fields (cooldown / blocked models) from DB.
		synth, _ := item["pool"].(map[string]any)
		if synth == nil {
			synth = map[string]any{}
			item["pool"] = synth
		}
		for _, k := range []string{"in_cooldown", "cooldown_until", "cooldown_code", "cooldown_model", "cooldown_reason", "blocked_models", "blocked_model_ids", "last_probe", "last_probe_status"} {
			if v, ok := poolView[k]; ok && v != nil {
				synth[k] = v
				item[k] = v
			}
		}
		// If DB already has the quota write, prefer its pool_status.
		if v, ok := poolView["pool_status"].(string); ok && v == "quota_disabled" {
			synth["pool_status"] = v
			synth["disabled_for_quota"] = true
			synth["enabled"] = false
			item["pool_status"] = v
			item["disabled_for_quota"] = true
			item["auto_disabled"] = true
			item["pool_disabled"] = true
		}
	}
	// Surface top-level fields used by frontend even when only synth pool exists.
	if pool, _ := item["pool"].(map[string]any); pool != nil {
		if v, ok := pool["pool_status"]; ok {
			item["pool_status"] = v
		}
		if v, ok := pool["disabled_for_quota"]; ok {
			item["disabled_for_quota"] = v
			if b, ok := v.(bool); ok && b {
				item["auto_disabled"] = true
				item["pool_disabled"] = true
			}
		}
		if v, ok := pool["enabled"]; ok {
			item["enabled"] = v
		}
		if v, ok := pool["in_cooldown"]; ok {
			item["in_cooldown"] = v
		}
	}
	writeJSON(w, http.StatusOK, item)
}

func serveIntegrationSettingsGet(w http.ResponseWriter, r *http.Request, options Options, key string) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "config": integrations.PublicConfig(r.Context(), options.Store, key)})
}

func serveIntegrationSettingsPut(w http.ResponseWriter, r *http.Request, options Options, key string) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	doTest := false
	if v, ok := body["test"].(bool); ok {
		doTest = v
		delete(body, "test")
	}
	cfg, err := integrations.SaveConfig(r.Context(), options.Store, key, body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	out := map[string]any{"ok": true, "config": cfg}
	if doTest {
		raw, _ := options.Store.GetSetting(r.Context(), key)
		rm, _ := raw.(map[string]any)
		switch key {
		case "cliproxyapi_config":
			test := integrations.TestCLIProxy(r.Context(), rm)
			out["test"] = test
			out["ok"] = test["ok"] == true
		case "sub2api_config":
			test := integrations.TestSub2API(r.Context(), rm)
			out["test"] = test
			out["ok"] = test["ok"] == true
			if v, ok := test["groups"]; ok {
				out["groups"] = v
			}
		}
	}
	writeJSON(w, http.StatusOK, out)
}

func serveCLIProxyTest(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	raw, err := options.Store.GetSetting(r.Context(), "cliproxyapi_config")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "config missing"})
		return
	}
	rm, _ := raw.(map[string]any)
	test := integrations.TestCLIProxy(r.Context(), rm)
	writeJSON(w, http.StatusOK, map[string]any{"ok": test["ok"] == true, "test": test})
}

func serveExportCLIProxyFormat(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	ids := stringSlice(body["ids"])
	if len(ids) == 0 {
		ids = stringSlice(body["account_ids"])
	}
	if body["all"] == true {
		ids = nil
	}
	out, err := integrations.ExportCLIProxyBundle(r.Context(), options.Store, ids)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func servePushCLIProxy(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	ids := stringSlice(body["account_ids"])
	if body["all"] == true || body["account_ids"] == nil {
		ids = nil
	}
	out, err := integrations.PushCLIProxy(r.Context(), options.Store, ids, 4)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func serveExportSub2APIFormat(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	ids := stringSlice(body["ids"])
	if len(ids) == 0 {
		ids = stringSlice(body["account_ids"])
	}
	if body["all"] == true {
		ids = nil
	}
	out, err := integrations.ExportSub2APIFormat(r.Context(), options.Store, ids)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func parseUpstreamModels(payload any) []map[string]any {
	var data []any
	switch p := payload.(type) {
	case map[string]any:
		if raw, ok := p["data"].([]any); ok {
			data = raw
		} else if raw, ok := p["models"].([]any); ok {
			data = raw
		} else if raw, ok := p["result"].([]any); ok {
			data = raw
		} else if nested, ok := p["data"].(map[string]any); ok {
			if raw, ok := nested["data"].([]any); ok {
				data = raw
			} else if raw, ok := nested["models"].([]any); ok {
				data = raw
			}
		}
	case []any:
		data = p
	}
	items := []map[string]any{}
	seen := map[string]bool{}
	for _, raw := range data {
		m, ok := raw.(map[string]any)
		if !ok {
			if s, ok := raw.(string); ok {
				s = strings.TrimSpace(s)
				if s == "" || seen[strings.ToLower(s)] {
					continue
				}
				seen[strings.ToLower(s)] = true
				items = append(items, map[string]any{"id": s, "name": s, "owned_by": "xai"})
			}
			continue
		}
		id := firstNonEmptyStr(stringValue(m["id"]), stringValue(m["model"]), stringValue(m["name"]))
		id = strings.TrimSpace(id)
		if id == "" || seen[strings.ToLower(id)] {
			continue
		}
		seen[strings.ToLower(id)] = true
		item := map[string]any{"id": id, "owned_by": firstNonEmptyStr(stringValue(m["owned_by"]), "xai")}
		if n := firstNonEmptyStr(stringValue(m["name"]), id); n != "" {
			item["name"] = n
		}
		if d := stringValue(m["description"]); d != "" {
			item["description"] = d
		}
		if cw, ok := m["context_window"]; ok {
			item["context_window"] = cw
		} else if cw, ok := m["context_length"]; ok {
			item["context_window"] = cw
		} else if cw, ok := m["max_context_length"]; ok {
			item["context_window"] = cw
		}
		if v, ok := m["supports_reasoning_effort"]; ok {
			item["supports_reasoning_effort"] = v
		}
		// keep useful extras without huge blobs
		for _, key := range []string{"max_completion_tokens", "reasoning_effort", "reasoning_efforts", "supported_in_api"} {
			if v, ok := m[key]; ok && v != nil {
				item[key] = v
			}
		}
		items = append(items, item)
	}
	return items
}

func ensureLocalModelExtras(items []map[string]any, defaultModel string) []map[string]any {
	if defaultModel == "" {
		defaultModel = "grok-4.5"
	}
	have := map[string]bool{}
	for _, it := range items {
		if id := strings.ToLower(strings.TrimSpace(stringValue(it["id"]))); id != "" {
			have[id] = true
		}
	}
	extras := []map[string]any{
		{"id": defaultModel, "name": defaultModel, "owned_by": "xai"},
		{"id": "grok-build", "name": "Grok Build", "description": "Grok coding / build model (cli-chat-proxy)", "owned_by": "xai", "synthetic": true},
		{"id": "grok-search", "name": "Grok Search", "description": "Grok with web search enabled (local alias)", "owned_by": "xai", "synthetic": true},
	}
	for _, ex := range extras {
		id := strings.ToLower(stringValue(ex["id"]))
		if id == "" || have[id] {
			continue
		}
		items = append(items, ex)
		have[id] = true
	}
	return items
}

func usageDetail(route string, requestBody map[string]any, ttftMS, latency int) map[string]any {
	detail := map[string]any{"route": route, "latency_ms": latency}
	if ttftMS > 0 {
		detail["ttft_ms"] = ttftMS
	}
	if effort := extractReasoningEffort(requestBody); effort != "" {
		detail["reasoning_effort"] = effort
		detail["thinking_intensity"] = effort
	}
	return detail
}

// extractReasoningEffort normalizes client thinking intensity to low/medium/high/xhigh.
// Codex: auto/default/standard/extra-high · Claude Code: low/medium/high/xhigh + budget_tokens.
func extractReasoningEffort(payload map[string]any) string {
	return reasoning.FromRequest(payload)
}

func normalizeReasoningEffort(value any) string {
	return reasoning.Normalize(value)
}

func budgetToEffort(n int) string {
	return reasoning.BudgetToLevel(n)
}

func firstNonEmptyStr(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

var (
	exportJobsMu sync.Mutex
	exportJobs   = map[string]map[string]any{}
)

func startExportJob(result map[string]any, filename string) map[string]any {
	jobID := fmt.Sprintf("exp_%d", time.Now().UnixNano())
	auth, _ := result["auth"].(map[string]any)
	count := 0
	if auth != nil {
		count = len(auth)
	} else if v, ok := result["count"].(int); ok {
		count = v
	}
	payload, _ := json.MarshalIndent(result, "", "  ")
	job := map[string]any{
		"ok": true, "job_id": jobID, "status": "done", "phase": "done",
		"message": fmt.Sprintf("导出完成：%d 个账号", count),
		"percent": 100, "done": count, "total": count, "count": count,
		"success": count, "fail": 0, "download_ready": true,
		"filename": filename, "payload": payload, "created_at": time.Now().Unix(),
	}
	exportJobsMu.Lock()
	if len(exportJobs) > 20 {
		exportJobs = map[string]map[string]any{}
	}
	exportJobs[jobID] = job
	exportJobsMu.Unlock()
	return map[string]any{
		"ok": true, "job_id": jobID, "status": "done", "message": job["message"],
		"total": count, "count": count, "filename": filename, "download_ready": true,
	}
}

func serveExportJobStatus(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	id := r.PathValue("job_id")
	exportJobsMu.Lock()
	job := exportJobs[id]
	exportJobsMu.Unlock()
	if job == nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "job not found"})
		return
	}
	out := map[string]any{}
	for k, v := range job {
		if k == "payload" {
			continue
		}
		out[k] = v
	}
	writeJSON(w, http.StatusOK, out)
}

func serveExportJobDownload(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	id := r.PathValue("job_id")
	exportJobsMu.Lock()
	job := exportJobs[id]
	exportJobsMu.Unlock()
	if job == nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"detail": "job not found"})
		return
	}
	payload, _ := job["payload"].([]byte)
	if len(payload) == 0 {
		writeJSON(w, http.StatusConflict, map[string]any{"detail": "export not ready"})
		return
	}
	filename, _ := job["filename"].(string)
	if filename == "" {
		filename = "grok2api-auth-export.json"
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filename))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(payload)
}

func servePushSub2API(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	ids := stringSlice(body["account_ids"])
	if len(ids) == 0 {
		ids = stringSlice(body["ids"])
	}
	if body["all"] == true {
		ids = nil
	}
	var groupID *int
	if v, ok := body["group_id"]; ok && v != nil {
		switch t := v.(type) {
		case float64:
			n := int(t)
			groupID = &n
		case json.Number:
			if i, err := t.Int64(); err == nil {
				n := int(i)
				groupID = &n
			}
		case string:
			if i, err := strconv.Atoi(strings.TrimSpace(t)); err == nil {
				groupID = &i
			}
		}
	}
	out, err := integrations.PushSub2API(r.Context(), options.Store, ids, groupID, 4)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error(), "ok": false})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func serveSub2APITest(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	raw, err := options.Store.GetSetting(r.Context(), "sub2api_config")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "config missing"})
		return
	}
	rm, _ := raw.(map[string]any)
	test := integrations.TestSub2API(r.Context(), rm)
	// Flatten groups / group_count for the settings page (testSub2apiConnection /
	// renderSub2apiGroups expect top-level fields, not only nested under "test").
	out := map[string]any{"ok": test["ok"] == true, "test": test}
	if v, ok := test["groups"]; ok {
		out["groups"] = v
	}
	if v, ok := test["group_count"]; ok {
		out["group_count"] = v
	}
	if v, ok := test["error"]; ok {
		out["error"] = v
	}
	if v, ok := test["message"]; ok {
		out["message"] = v
	}
	writeJSON(w, http.StatusOK, out)
}

func serveSub2APIGroupsList(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, false) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	out := integrations.ListSub2APIGroups(r.Context(), options.Store)
	// Keep HTTP 200 with ok:false for remote/config errors so the admin toast
	// can show the message; only auth/store failures use non-200 above.
	writeJSON(w, http.StatusOK, out)
}

func serveSub2APIGroupsCreate(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	name := stringValue(body["name"])
	platform := stringValue(body["platform"])
	if platform == "" {
		platform = "grok"
	}
	setDefault := true
	if v, ok := body["set_default"].(bool); ok {
		setDefault = v
	}
	out := integrations.CreateSub2APIGroup(r.Context(), options.Store, name, platform, setDefault)
	writeJSON(w, http.StatusOK, out)
}

func buildSSOExport(authMap map[string]any) map[string]any {
	auth, _ := authMap["auth"].(map[string]any)
	lines := []string{}
	items := []map[string]any{}
	includePassword := false
	for id, raw := range auth {
		entry, _ := raw.(map[string]any)
		if entry == nil {
			continue
		}
		sso := accounts.GetSSOValue(entry)
		if sso == "" {
			continue
		}
		email := ""
		if v, ok := entry["email"].(string); ok {
			email = strings.TrimSpace(v)
		}
		password := ""
		if v, ok := entry["password"].(string); ok {
			password = strings.TrimSpace(v)
		}
		if password == "" {
			if v, ok := entry["register_password"].(string); ok {
				password = strings.TrimSpace(v)
			}
		}
		line := sso
		if email != "" && password != "" {
			line = email + "----" + sso + "----" + password
			includePassword = true
		} else if email != "" {
			line = email + "----" + sso
		}
		lines = append(lines, line)
		items = append(items, map[string]any{"id": id, "email": email, "sso": sso})
	}
	content := strings.Join(lines, "\n")
	return map[string]any{
		"ok":               true,
		"count":            len(items),
		"with_sso":         len(items),
		"lines":            lines,
		"items":            items,
		"text":             content,
		"content":          content,
		"format":           "txt",
		"ext":              "txt",
		"media_type":       "text/plain;charset=utf-8",
		"filename":         "grok2api-accounts-sso.txt",
		"include_password": includePassword,
	}
}

type statusProvider interface{ Status() map[string]any }

func serviceStatus(svc statusProvider, options Options) map[string]any {
	if svc == nil {
		return map[string]any{"enabled": false, "implementation": "go", "started": false}
	}
	switch v := any(svc).(type) {
	case *maintainer.Service:
		if v == nil {
			return map[string]any{"enabled": false, "implementation": "go", "started": false}
		}
	case *modelhealth.Service:
		if v == nil {
			return map[string]any{"enabled": false, "implementation": "go", "started": false}
		}
	}
	return svc.Status()
}

func serveAdminProbeAccount(w http.ResponseWriter, r *http.Request, options Options) {
	if !requireAdminReadWrite(w, r, options, true) {
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	model := stringValue(body["model"])
	if model == "" {
		model = options.runtimeConfig().DefaultModel
	}
	model = modelCatalog(options).Resolve(model)
	accountID := strings.TrimSpace(r.PathValue("account_id"))
	if accountID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "account_id required"})
		return
	}
	// Manual "模型测试" is full 测活: last_probe + kick/block/clear must land in DB
	// (same semantics as probe-batch / background model health).
	autoDisable := true
	if v, ok := body["auto_disable"].(bool); ok {
		autoDisable = v
	}
	if options.ModelHealth != nil {
		results := options.ModelHealth.ProbeIDs(r.Context(), []string{accountID}, model, autoDisable, "manual")
		if len(results) == 0 {
			writeJSON(w, http.StatusNotFound, map[string]any{"detail": "account not found"})
			return
		}
		row := results[0]
		// ProbeIDs shape: {ok, account_id, email, result:probe} OR {ok:false, account_id, error}
		result, _ := row["result"].(map[string]any)
		if result == nil {
			result = map[string]any{
				"account_id": firstNonEmptyStr(stringValue(row["account_id"]), accountID),
				"model":      model,
				"available":  false,
				"ok":         false,
				"error":      firstNonEmptyStr(stringValue(row["error"]), "probe produced no result"),
				"source":     "manual",
				"probed_at":  time.Now().Unix(),
			}
		}
		aid := firstNonEmptyStr(stringValue(row["account_id"]), stringValue(result["account_id"]), accountID)
		email := firstNonEmptyStr(stringValue(row["email"]), stringValue(result["email"]))
		// ProbeIDs already deferred last_probe batch save; do not block the admin
		// response on another synchronous PG write. Kick/Clear already applied.
		ok := row["ok"] == true || result["available"] == true
		// Prefer live probe flags for immediate UI; enrich with DB pool if present.
		poolView := map[string]any{
			"id": aid, "account_id": aid, "last_probe": result,
			"last_probe_status": map[bool]string{true: "ok", false: "fail"}[ok],
		}
		if ok {
			poolView["pool_status"] = "normal"
			poolView["in_cooldown"] = false
			if result["recovered"] == true {
				poolView["recovered"] = true
			}
		} else {
			if result["kicked_cooldown"] == true {
				poolView["in_cooldown"] = true
				poolView["pool_status"] = "cooldown"
				poolView["cooldown_code"] = result["cooldown_code"]
			}
			if result["model_blocked"] == true {
				poolView["pool_status"] = "model_blocked"
			}
			if result["auto_disabled"] == true {
				poolView["enabled"] = false
				poolView["pool_status"] = "disabled"
			}
		}
		if dbPool, perr := options.Store.GetAccountPoolView(r.Context(), aid); perr == nil && dbPool != nil {
			// Merge durable fields without overwriting live last_probe snapshot.
			for k, v := range dbPool {
				if k == "last_probe" {
					continue
				}
				if _, has := poolView[k]; !has || poolView[k] == nil {
					poolView[k] = v
				}
			}
		}
		statusCode := 0
		switch v := result["status_code"].(type) {
		case int:
			statusCode = v
		case int64:
			statusCode = int(v)
		case float64:
			statusCode = int(v)
		case json.Number:
			if n, err := v.Int64(); err == nil {
				statusCode = int(n)
			}
		}
		errText := stringValue(result["error"])
		if ok {
			touchRedisPool(options, aid, true, "", nil, statusCode)
		} else {
			touchRedisPool(options, aid, false, errText, nil, statusCode)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"ok": ok, "account_id": aid, "email": email,
			"result": result, "pool": poolView,
		})
		return
	}
	// Fallback when ModelHealth is not wired (tests / degraded): still persist last_probe.
	auth, err := options.Store.GetAccountAuth(r.Context(), accountID)
	if err != nil || auth == nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"detail": "account not found or has no token"})
		return
	}
	client := upstreamClient(options)
	probeBody := map[string]any{
		"model": model, "stream": true, "max_tokens": 1,
		"messages": []any{map[string]any{"role": "user", "content": "ping"}},
	}
	started := time.Now()
	resp, err := client.Open(r.Context(), grok.Account{ID: auth.ID, Token: auth.Token}, model, probeBody)
	result := map[string]any{
		"account_id": auth.ID, "email": auth.Email, "model": model,
		"probed_at": time.Now().Unix(), "source": "manual",
	}
	if err != nil {
		status := 0
		errText := err.Error()
		var ue *grok.UpstreamError
		if errors.As(err, &ue) {
			status = ue.Status
			errText = ue.Body
			if len(errText) > 400 {
				errText = errText[:400]
			}
		}
		if autoDisable {
			decision := pool.ClassifyUpstreamFailure(status, errText, model)
			if status == 401 || status == 403 {
				_, _ = options.Store.SetAccountEnabled(r.Context(), auth.ID, false)
				result["auto_disabled"] = true
			} else if decision.ShouldCooldown {
				until := time.Now().Add(10 * time.Minute)
				if decision.Until != nil {
					until = *decision.Until
				}
				if decision.BlockModel {
					bm := model
					if decision.Model != "" {
						bm = decision.Model
					}
					_ = options.Store.BlockPoolModel(r.Context(), auth.ID, bm, &until)
					result["model_blocked"] = true
					result["blocked_model"] = bm
				}
				sec := until.Sub(time.Now()).Seconds()
				if sec < 60 {
					sec = 60
				}
				_, _ = options.Store.KickFromPool(r.Context(), auth.ID, errText, &sec)
				result["kicked_cooldown"] = true
			}
		}
		result["available"] = false
		result["ok"] = false
		result["error"] = errText
		result["status_code"] = status
		result["latency_ms"] = time.Since(started).Milliseconds()
		_ = options.Store.SaveLastProbe(r.Context(), auth.ID, result)
		poolView, _ := options.Store.GetAccountPoolView(r.Context(), auth.ID)
		touchRedisPool(options, auth.ID, false, errText, nil, status)
		writeJSON(w, http.StatusOK, map[string]any{"ok": false, "account_id": auth.ID, "email": auth.Email, "result": result, "pool": poolView})
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1024))
	_ = resp.Body.Close()
	_ = options.Store.ReportPoolSuccess(r.Context(), auth.ID, true)
	if _, err := options.Store.ClearAccountCooldown(r.Context(), auth.ID); err == nil {
		result["recovered"] = true
	}
	if model != "" {
		if err := options.Store.UnblockPoolModel(r.Context(), auth.ID, model); err == nil {
			result["unblocked_model"] = model
		}
	}
	result["available"] = true
	result["ok"] = true
	result["status_code"] = resp.StatusCode
	result["latency_ms"] = time.Since(started).Milliseconds()
	_ = options.Store.SaveLastProbe(r.Context(), auth.ID, result)
	touchRedisPool(options, auth.ID, true, "", nil, resp.StatusCode)
	poolView, _ := options.Store.GetAccountPoolView(r.Context(), auth.ID)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "account_id": auth.ID, "email": auth.Email, "result": result, "pool": poolView})
}

func serveAdminSetAccountStatus(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var body map[string]any
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	_ = decoder.Decode(&body)
	if body == nil {
		body = map[string]any{}
	}
	status := strings.TrimSpace(stringValue(body["status"]))
	if status == "" {
		status = strings.TrimSpace(stringValue(body["pool_status"]))
	}
	if status == "" {
		status = strings.TrimSpace(stringValue(body["tag"]))
	}
	if status == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"detail":  "status required",
			"allowed": []string{"live", "cooldown", "model_blocked", "quota_disabled", "disabled", "expired"},
		})
		return
	}
	reason := strings.TrimSpace(stringValue(body["reason"]))
	if reason == "" {
		reason = strings.TrimSpace(stringValue(body["message"]))
	}
	model := strings.TrimSpace(stringValue(body["model"]))
	if model == "" {
		model = strings.TrimSpace(stringValue(body["blocked_model"]))
	}
	var cooldown *float64
	for _, key := range []string{"cooldown_sec", "cooldown_seconds", "sec"} {
		switch v := body[key].(type) {
		case float64:
			cooldown = &v
		case json.Number:
			if f, err := v.Float64(); err == nil {
				cooldown = &f
			}
		}
		if cooldown != nil {
			break
		}
	}
	rec, err := options.Store.SetAccountPoolStatus(r.Context(), r.PathValue("account_id"), status, reason, model, cooldown)
	if err != nil {
		if postgres.IsAccountNotFound(err) {
			writeJSON(w, http.StatusNotFound, map[string]any{"detail": "Account not found"})
			return
		}
		msg := err.Error()
		if strings.HasPrefix(msg, "unsupported status") {
			writeJSON(w, http.StatusBadRequest, map[string]any{"detail": msg})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": msg})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "account": rec, "pool": rec, "status": rec["pool_status"]})
}

func serveAdminKickAccount(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	reason := stringValue(body["reason"])
	var cooldown *float64
	switch v := body["cooldown_sec"].(type) {
	case float64:
		cooldown = &v
	case json.Number:
		if f, err := v.Float64(); err == nil {
			cooldown = &f
		}
	}
	rec, err := options.Store.KickFromPool(r.Context(), r.PathValue("account_id"), reason, cooldown)
	if err != nil {
		if postgres.IsAccountNotFound(err) {
			writeJSON(w, http.StatusNotFound, map[string]any{"detail": "account not found"})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "account": rec})
}

func serveAdminClearCooldown(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	rec, err := options.Store.ClearAccountCooldown(r.Context(), r.PathValue("account_id"))
	if err != nil {
		if postgres.IsAccountNotFound(err) {
			writeJSON(w, http.StatusNotFound, map[string]any{"detail": "account not found"})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "account": rec})
	if options.Redis != nil {
		_ = options.Redis.MirrorCooldown(r.Context(), r.PathValue("account_id"), time.Time{})
		_, _ = options.Redis.TouchStats(r.Context(), r.PathValue("account_id"), redis.PoolStatsTouch{Success: true, ClearCooldown: true})
	}
}

func adminWriteAllowed(w http.ResponseWriter, r *http.Request, options Options) bool {
	if !options.AdminWriteEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin write routes are not enabled"})
		return false
	}
	if !options.AdminReadEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin read routes are not enabled"})
		return false
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return false
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return false
	}
	return true
}

func serveAdminSetup(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	password := stringValue(body["password"])
	if len(password) < 4 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "password must contain at least 4 characters"})
		return
	}
	has, err := options.Store.HasAdminPassword(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if has {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "admin password already configured"})
		return
	}
	hash, salt, err := adminauth.NewPassword(password)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	if err := options.Store.SetAdminPassword(r.Context(), hash, salt); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	token, err := createAdminSession(options)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	setAdminCookie(w, token)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "token": token, "message": "Admin password created"})
}

func serveAdminLogin(w http.ResponseWriter, r *http.Request, options Options) {
	if !options.AdminReadEnabled && !options.AdminWriteEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin routes are not enabled"})
		return
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return
	}
	if options.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "PostgreSQL store unavailable"})
		return
	}
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	password := stringValue(body["password"])
	ok, err := verifyAdminPassword(r.Context(), options, password)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	if !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Invalid admin password"})
		return
	}
	token, err := createAdminSession(options)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	setAdminCookie(w, token)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "token": token})
}

func serveAdminSession(w http.ResponseWriter, r *http.Request, options Options) {
	if !options.AdminReadEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin read routes are not enabled"})
		return
	}
	if !isReady(options) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": readyReason(options)})
		return
	}
	token, ok := admin.RequireSession(r, options.AdminSessions)
	if !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"ok": false, "authenticated": false})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "authenticated": true, "token": token})
}

func serveAdminLogout(w http.ResponseWriter, r *http.Request, options Options) {
	if !options.AdminReadEnabled && !options.AdminWriteEnabled {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"detail": "Go admin routes are not enabled"})
		return
	}
	token := admin.ExtractSession(r)
	if token != "" {
		deleteAdminSession(options, token)
	}
	clearAdminCookie(w)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func serveAdminCreateKey(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	name := stringValue(body["name"])
	note := stringValue(body["note"])
	result, err := options.Store.CreateAPIKey(r.Context(), name, note)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"detail": err.Error()})
		return
	}
	payload := result.Record.PublicMap()
	payload["secret"] = result.Secret
	payload["key"] = result.Secret
	writeJSON(w, http.StatusOK, payload)
}

func serveAdminUpdateKey(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	id := r.PathValue("key_id")
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	var name, note *string
	var enabled *bool
	if v, ok := body["name"].(string); ok {
		name = &v
	}
	if v, ok := body["note"].(string); ok {
		note = &v
	}
	if v, ok := body["enabled"].(bool); ok {
		enabled = &v
	}
	rec, err := options.Store.UpdateAPIKey(r.Context(), id, name, note, enabled)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, rec.PublicMap())
}

func serveAdminRegenerateKey(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	result, err := options.Store.RegenerateAPIKey(r.Context(), r.PathValue("key_id"))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	payload := result.Record.PublicMap()
	payload["secret"] = result.Secret
	payload["key"] = result.Secret
	writeJSON(w, http.StatusOK, payload)
}

func serveAdminDeleteKey(w http.ResponseWriter, r *http.Request, options Options) {
	if !adminWriteAllowed(w, r, options) {
		return
	}
	if _, ok := admin.RequireSession(r, options.AdminSessions); !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"detail": "Admin authentication required"})
		return
	}
	ok, err := options.Store.DeleteAPIKey(r.Context(), r.PathValue("key_id"))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]any{"detail": "api key not found"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func verifyAdminPassword(ctx context.Context, options Options, password string) (bool, error) {
	if options.Store == nil {
		return false, errors.New("store unavailable")
	}
	pw, err := options.Store.LoadAdminPassword(ctx)
	if err == nil && pw.Hash != "" && pw.Salt != "" {
		return adminauth.VerifyPassword(password, pw.Hash, pw.Salt), nil
	}
	// bootstrap via env password only when no store hash exists
	envPW := strings.TrimSpace(options.Config.LegacyAdminPassword)
	if envPW == "" {
		// fallback common env already loaded? use os.Getenv for ADMIN_PASSWORD
		envPW = strings.TrimSpace(os.Getenv("GROK2API_ADMIN_PASSWORD"))
	}
	if envPW != "" && subtle.ConstantTimeCompare([]byte(password), []byte(envPW)) == 1 {
		return true, nil
	}
	return false, nil
}

func createAdminSession(options Options) (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	token := base64.RawURLEncoding.EncodeToString(buf)
	// Prefer Redis session store (Python path), fall back to Postgres sessions map.
	if rc, ok := options.AdminSessions.(interface{ CreateAdminSession(string) error }); ok {
		if err := rc.CreateAdminSession(token); err == nil {
			return token, nil
		}
	}
	if options.Store != nil {
		if err := options.Store.CreateAdminSession(token); err != nil {
			return "", err
		}
		return token, nil
	}
	return "", errors.New("no admin session store available")
}

func deleteAdminSession(options Options, token string) {
	if rc, ok := options.AdminSessions.(interface{ DeleteAdminSession(string) error }); ok {
		_ = rc.DeleteAdminSession(token)
	}
	if options.Store != nil {
		_ = options.Store.DeleteAdminSession(token)
	}
}

func setAdminCookie(w http.ResponseWriter, token string) {
	http.SetCookie(w, &http.Cookie{
		Name:     admin.AdminCookie,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   int((7 * 24 * time.Hour).Seconds()),
	})
}

func clearAdminCookie(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{Name: admin.AdminCookie, Value: "", Path: "/", MaxAge: -1, HttpOnly: true})
}

func serveAdminPage(w http.ResponseWriter, r *http.Request, staticDir, page string) {
	name := strings.TrimSpace(strings.TrimSuffix(page, ".html"))
	if name == "" {
		name = "index"
	}
	if !allowedAdminPage(name) {
		http.NotFound(w, r)
		return
	}
	serveFile(w, r, filepath.Join(staticDir, "admin", name+".html"), true)
}

func allowedAdminPage(name string) bool {
	switch name {
	case "index", "overview", "login", "keys", "accounts", "models", "guide", "settings", "logs", "usage":
		return true
	default:
		return false
	}
}

func serveStatic(w http.ResponseWriter, r *http.Request, staticDir, name string) {
	cleaned := filepath.Clean("/" + name)
	if cleaned == "/" || strings.Contains(cleaned, "..") {
		http.NotFound(w, r)
		return
	}
	// Content-hashed dist assets (core.f5e0a3148a.js) are immutable forever.
	// HTML admin pages stay no-store via serveAdminPage. Non-hashed sources
	// (static/js/*.js) are not long-cached so deploys without hash don't stick.
	longCache := isHashedDistAsset(cleaned)
	serveFile(w, r, filepath.Join(staticDir, cleaned), false, longCache)
}

// isHashedDistAsset reports whether cleaned path is under /dist/ and the
// basename looks like name.<8+hex>.js|css (content-hash fingerprint).
func isHashedDistAsset(cleaned string) bool {
	// cleaned is like "/dist/core.f5e0a3148a.js"
	if !strings.HasPrefix(cleaned, "/dist/") {
		return false
	}
	base := filepath.Base(cleaned)
	// e.g. core.f5e0a3148a.js, admin-antd.1031c5bb2f.css
	dot := strings.LastIndex(base, ".")
	if dot <= 0 {
		return false
	}
	ext := strings.ToLower(base[dot+1:])
	if ext != "js" && ext != "css" {
		return false
	}
	name := base[:dot]
	hashDot := strings.LastIndex(name, ".")
	if hashDot <= 0 || hashDot == len(name)-1 {
		return false
	}
	hash := name[hashDot+1:]
	if len(hash) < 8 {
		return false
	}
	for _, c := range hash {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

func serveFile(w http.ResponseWriter, r *http.Request, name string, noStore bool, longCache ...bool) {
	if noStore {
		w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
		w.Header().Set("Pragma", "no-cache")
		w.Header().Set("Expires", "0")
	} else if len(longCache) > 0 && longCache[0] {
		// Content-hashed dist filenames are safe to cache for 1y.
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	}
	info, err := os.Stat(name)
	if err != nil || info.IsDir() {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, name)
}

func isReady(options Options) bool {
	return options.Ready != nil && options.Ready()
}

func readyReason(options Options) string {
	if options.Reason == nil {
		return "not ready"
	}
	return options.Reason()
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	// Marshal first so encode failures never produce HTTP 200 + empty body
	// (which the admin UI then treats as null and crashes on field access).
	payload, err := json.Marshal(value)
	if err != nil {
		slog.Error("writeJSON marshal failed", "error", err)
		payload, _ = json.Marshal(map[string]any{"ok": false, "detail": "encode response failed"})
		status = http.StatusInternalServerError
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(payload)
	if len(payload) == 0 || payload[len(payload)-1] != '\n' {
		_, _ = w.Write([]byte("\n"))
	}
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	return "1"
}

func itoaPort(value int) string {
	return strconv.Itoa(value)
}

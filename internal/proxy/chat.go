package proxy

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/hm2899/grokcli-2api/internal/models"
	"github.com/hm2899/grokcli-2api/internal/pool"
	"github.com/hm2899/grokcli-2api/internal/protocol/anthropic"
	"github.com/hm2899/grokcli-2api/internal/protocol/toolcall"
	"github.com/hm2899/grokcli-2api/internal/upstream/grok"
)

// AccountFailureReporter is notified for every upstream account attempt that
// failed (including intermediate failover losers). Server wiring uses this to
// classify free-usage / 额度用完 bodies and kick accounts into the cooldown pool
// even when a later account eventually succeeds the request.
type AccountFailureReporter interface {
	ReportAccountFailure(accountID, model string, err error)
}

type ChatService struct {
	Catalog       *models.Catalog
	Client        *grok.Client
	Now           func() time.Time
	PickObserver  PickObserver
	AffinityStore AffinityStore
	// FailureReporter is optional; when set, every failed account attempt is
	// reported so quota-exhausted text can enter the cooldown pool immediately.
	FailureReporter       AccountFailureReporter
	StickyFirstOnly       bool // try sticky/first account before broader failover
	FirstByteProbeWorkers int  // parallel first-byte probes after sticky miss (default 3, max 8)
}

type PickObserver interface {
	LoadPenalty(context.Context, string) int64
	MarkPick(context.Context, string)
	ReleasePick(context.Context, string)
}

// optional batching extension for hot-path candidate windows
type batchPickObserver interface {
	LoadPenalties(context.Context, []string) map[string]int64
}

type AffinityStore interface {
	GetAffinity(context.Context, string) (string, error)
	BindAffinity(context.Context, string, string) error
	// ClearAffinity drops a multi-turn pin (dead/cooling sticky account).
	ClearAffinity(context.Context, string) error
}

type ChatRequest struct {
	Model     string         `json:"model"`
	Stream    bool           `json:"stream"`
	Raw       map[string]any `json:"-"`
	UserAgent string         `json:"-"` // optional; Codex auto-compact threshold
}

type StreamFrame struct {
	Data []byte
	Done bool
}

type ChatDelta struct {
	ID           string
	Model        string
	Created      int64
	Content      string
	Reasoning    string
	ToolCalls    []map[string]any
	FunctionCall map[string]any
	FinishReason any
	Usage        any
}

type ChatResult struct {
	Payload       map[string]any
	AccountID     string
	Model         string
	Usage         any
	PreferAccount string
	FirstAccount  string
	Failover      bool
	Fingerprint   string
	Accounts      int
	Prep          BodyPrepStats
}

type StreamOpen struct {
	Body          io.ReadCloser
	AccountID     string
	Model         string
	PreferAccount string
	FirstAccount  string
	Failover      bool
	Fingerprint   string
	Accounts      int
	Prep          BodyPrepStats
}

type StreamStats struct {
	Usage        any
	FirstTokenMS int // 0 if never observed
}

func DecodeChatRequest(reader io.Reader) (ChatRequest, error) {
	var raw map[string]any
	decoder := json.NewDecoder(reader)
	decoder.UseNumber()
	if err := decoder.Decode(&raw); err != nil {
		return ChatRequest{}, err
	}
	model, _ := raw["model"].(string)
	stream, _ := raw["stream"].(bool)
	return ChatRequest{Model: model, Stream: stream, Raw: raw}, nil
}

func (s *ChatService) Complete(ctx context.Context, request ChatRequest, candidates []pool.Candidate, mode string) (map[string]any, error) {
	result, err := s.CompleteWithResult(ctx, request, candidates, mode)
	if err != nil {
		return nil, err
	}
	return result.Payload, nil
}

func (s *ChatService) CompleteWithResult(ctx context.Context, request ChatRequest, candidates []pool.Candidate, mode string) (ChatResult, error) {
	if request.Stream {
		return ChatResult{}, fmt.Errorf("Go chat streaming requires ChatService.Stream")
	}
	model, chain, client, err := s.prepareChain(ctx, request, candidates, mode)
	if err != nil {
		return ChatResult{}, err
	}
	accounts := upstreamAccounts(chain)
	body, prep := PrepareUpstreamBodyDetailed(request.Raw, request.UserAgent)
	fingerprint := ChatFingerprint(request)
	// Prefer account already boosted to chain[0] by prepareChain/ensureStickyCandidate.
	// Avoid a second Redis GET on the TTFT hot path.
	prefer := ""
	if len(chain) > 0 {
		prefer = chain[0].ID
	}
	first := ""
	if len(accounts) > 0 {
		first = accounts[0].ID
	}
	var lastEmpty error
	lastFailAccountID := ""
	for i, account := range accounts {
		s.markAttempt(ctx, account.ID)
		attempt, err := OpenWithFailover(ctx, client, []grok.Account{account}, model, body, &CommitState{})
		if err != nil {
			// Intermediate + final losers: report so free-usage / 额度用完 bodies
			// enter the cooldown pool even when a later account succeeds.
			s.reportAccountFailure(account.ID, model, err)
			lastFailAccountID = account.ID
			// Retryable/non-retryable both continue within short chain until exhausted.
			if i == len(accounts)-1 {
				s.releaseChain(ctx, chain)
				if lastEmpty != nil {
					return ChatResult{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, AccountID: lastFailAccountID}, lastEmpty
				}
				return ChatResult{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, AccountID: lastFailAccountID}, err
			}
			continue
		}
		collector := newChatCollector(model)
		collector.SetAllowedTools(extractAllowedToolNames(request.Raw))
		readErr := grok.ReadSSE(attempt.Body, collector.feed)
		_ = attempt.Body.Close()
		if readErr != nil {
			s.reportAccountFailure(attempt.Account.ID, model, readErr)
			s.releaseChain(ctx, chain)
			return ChatResult{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, AccountID: attempt.Account.ID}, readErr
		}
		if !collector.emptyModelOutput() {
			s.releaseChainExcept(ctx, chain, attempt.Account.ID)
			// Failover accounts must NOT steal the sticky pin — otherwise multi-turn
			// cannot return to the cache-warm primary after cooldown ends.
			failover := first != "" && attempt.Account.ID != first
			// Only skip rebinding when we failed over away from an EXISTING sticky pin.
			// Always pin the account that produced a live response so multi-turn
			// prompt-cache stays on the healthy account after sticky failover.
			s.bindAffinity(ctx, request, attempt.Account.ID)
			return ChatResult{
				Payload: collector.response(), AccountID: attempt.Account.ID, Model: collector.model, Usage: collector.usage,
				PreferAccount: prefer, FirstAccount: first, Failover: failover,
				Fingerprint: fingerprint, Accounts: len(chain), Prep: prep,
			}, nil
		}
		lastEmpty = &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
		s.reportAccountFailure(attempt.Account.ID, model, lastEmpty)
		lastFailAccountID = attempt.Account.ID
	}
	s.releaseChain(ctx, chain)
	if lastEmpty == nil {
		lastEmpty = &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
	}
	return ChatResult{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, Failover: true, AccountID: lastFailAccountID}, lastEmpty
}

func (s *ChatService) Stream(ctx context.Context, request ChatRequest, candidates []pool.Candidate, mode string, emit func(StreamFrame) error) error {
	body, err := s.OpenStream(ctx, request, candidates, mode)
	if err != nil {
		return err
	}
	defer body.Close()
	return ForwardChatStream(body, emit)
}

func (s *ChatService) OpenStream(ctx context.Context, request ChatRequest, candidates []pool.Candidate, mode string) (io.ReadCloser, error) {
	opened, err := s.OpenStreamWithResult(ctx, request, candidates, mode)
	if err != nil {
		return nil, err
	}
	return opened.Body, nil
}

func (s *ChatService) OpenStreamWithResult(ctx context.Context, request ChatRequest, candidates []pool.Candidate, mode string) (StreamOpen, error) {
	model, chain, client, err := s.prepareChain(ctx, request, candidates, mode)
	if err != nil {
		return StreamOpen{}, err
	}
	accounts := upstreamAccounts(chain)
	body, prep := PrepareUpstreamBodyDetailed(request.Raw, request.UserAgent)
	fingerprint := ChatFingerprint(request)
	// Prefer account already boosted to chain[0] by prepareChain/ensureStickyCandidate.
	// Avoid a second Redis GET on the TTFT hot path.
	prefer := ""
	if len(chain) > 0 {
		prefer = chain[0].ID
	}
	first := ""
	if len(accounts) > 0 {
		first = accounts[0].ID
	}
	// Sticky-first: try preferred/first account alone before spending TTFT on failover chain.
	stickyFirst := s.StickyFirstOnly
	if !stickyFirst {
		stickyFirst = prefer != "" && first != "" && prefer == first && len(accounts) > 1
	}

	meta := StreamOpen{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep}
	var lastEmpty error

	// openOne tries a single account: dial + empty-stream peek.
	// On success returns a live guarded body. On empty/error releases pick and returns ok=false.
	openOne := func(account grok.Account) (StreamOpen, bool, error) {
		s.markAttempt(ctx, account.ID)
		attempt, err := OpenWithFailover(ctx, client, []grok.Account{account}, model, body, &CommitState{})
		if err != nil {
			s.reportAccountFailure(account.ID, model, err)
			s.releasePick(ctx, account.ID)
			return StreamOpen{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, AccountID: account.ID}, false, err
		}
		guarded, empty, err := guardStreamAgainstEmpty(attempt.Body)
		if err != nil {
			_ = attempt.Body.Close()
			s.reportAccountFailure(attempt.Account.ID, model, err)
			s.releasePick(ctx, account.ID)
			return StreamOpen{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, AccountID: attempt.Account.ID}, false, err
		}
		if empty {
			_ = guarded.Close()
			s.releasePick(ctx, account.ID)
			emptyErr := &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
			s.reportAccountFailure(attempt.Account.ID, model, emptyErr)
			return StreamOpen{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, AccountID: attempt.Account.ID}, false, emptyErr
		}
		failover := first != "" && attempt.Account.ID != first
		// Always pin the live stream account (rebind after sticky failover).
		s.bindAffinity(ctx, request, attempt.Account.ID)
		return StreamOpen{
			Body: guarded, AccountID: attempt.Account.ID, Model: model,
			PreferAccount: prefer, FirstAccount: first, Failover: failover,
			Fingerprint: fingerprint, Accounts: len(chain), Prep: prep,
		}, true, nil
	}

	// Phase 1: sticky / first account alone (preserves prompt cache warmth).
	primary := accounts
	rest := []grok.Account(nil)
	if stickyFirst && len(accounts) > 1 {
		primary = accounts[:1]
		rest = accounts[1:]
	}
	for _, account := range primary {
		opened, ok, err := openOne(account)
		if ok {
			s.releaseChainExcept(ctx, chain, opened.AccountID)
			return opened, nil
		}
		// Sticky primary missed — drop the pin so the failover winner (and next
		// turn) does not keep routing to a dead/cooling account.
		if stickyFirst {
			s.noteStickyOutcome(ctx, request, account.ID, false)
		}
		if err != nil {
			// Prefer the concrete failing account id so open-failure paths still
			// feed reportChatPool / cooldown classification.
			if strings.TrimSpace(opened.AccountID) != "" {
				meta.AccountID = opened.AccountID
			} else {
				meta.AccountID = account.ID
			}
			if ue, is := err.(*grok.UpstreamError); is && strings.Contains(ue.Body, "empty model output") {
				lastEmpty = err
				continue
			}
			// Non-empty errors: still try rest if sticky-first; else fall through.
			if !(stickyFirst && len(rest) > 0) {
				s.releaseChain(ctx, chain)
				if lastEmpty != nil {
					return meta, lastEmpty
				}
				return meta, err
			}
			lastEmpty = err
		}
	}

	// Phase 2: parallel first-byte probe on remaining failover candidates.
	// Race dials; first non-empty stream wins. Others are closed immediately.
	// Caps concurrency to keep upstream load bounded (default chain is 4).
	if len(rest) == 0 {
		rest = nil
		// If stickyFirst was false we already tried all in primary.
		if !(stickyFirst && len(accounts) > 1) {
			s.releaseChain(ctx, chain)
			if lastEmpty == nil {
				lastEmpty = &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
			}
			return StreamOpen{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, Failover: true}, lastEmpty
		}
	}
	if len(rest) > 0 {
		opened, err := s.parallelFirstByteOpen(ctx, rest, client, model, body, chain, prefer, first, fingerprint, prep, request)
		if err == nil {
			return opened, nil
		}
		if lastEmpty == nil {
			lastEmpty = err
		}
	}

	s.releaseChain(ctx, chain)
	if lastEmpty == nil {
		lastEmpty = &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
	}
	return StreamOpen{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, Failover: true}, lastEmpty
}

// parallelFirstByteOpen races remaining failover accounts for the first non-empty
// upstream SSE. Sticky/primary is NOT included — call this only after sticky miss.
// Losers are closed promptly so we do not burn quota on multiple full generations.
func (s *ChatService) parallelFirstByteOpen(
	ctx context.Context,
	accounts []grok.Account,
	client *grok.Client,
	model string,
	body map[string]any,
	chain []pool.Candidate,
	prefer, first, fingerprint string,
	prep BodyPrepStats,
	request ChatRequest,
) (StreamOpen, error) {
	type raced struct {
		opened StreamOpen
		err    error
		idx    int
	}
	if len(accounts) == 0 {
		return StreamOpen{}, &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
	}
	// Bound parallel probes (configurable; default 3, hard max 8).
	maxWorkers := 3
	if s != nil && s.FirstByteProbeWorkers > 0 {
		maxWorkers = s.FirstByteProbeWorkers
	}
	if maxWorkers > 8 {
		maxWorkers = 8
	}
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	n := len(accounts)
	if n > maxWorkers {
		n = maxWorkers
		accounts = accounts[:maxWorkers]
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ch := make(chan raced, n)
	var wg sync.WaitGroup
	for i, account := range accounts {
		wg.Add(1)
		go func(i int, account grok.Account) {
			defer wg.Done()
			s.markAttempt(ctx, account.ID)
			attempt, err := OpenWithFailover(ctx, client, []grok.Account{account}, model, body, &CommitState{})
			if err != nil {
				// Parallel loser / exhausted account — still classify free-usage text.
				s.reportAccountFailure(account.ID, model, err)
				s.releasePick(ctx, account.ID)
				select {
				case ch <- raced{err: err, idx: i}:
				case <-ctx.Done():
				}
				return
			}
			// If we already lost the race, close immediately without peeking.
			select {
			case <-ctx.Done():
				_ = attempt.Body.Close()
				s.releasePick(context.Background(), account.ID)
				return
			default:
			}
			guarded, empty, err := guardStreamAgainstEmpty(attempt.Body)
			if err != nil {
				_ = attempt.Body.Close()
				s.reportAccountFailure(account.ID, model, err)
				s.releasePick(ctx, account.ID)
				select {
				case ch <- raced{err: err, idx: i}:
				case <-ctx.Done():
				}
				return
			}
			if empty {
				_ = guarded.Close()
				emptyErr := &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
				s.reportAccountFailure(account.ID, model, emptyErr)
				s.releasePick(ctx, account.ID)
				select {
				case ch <- raced{err: emptyErr, idx: i}:
				case <-ctx.Done():
				}
				return
			}
			// Winner candidate — hand off body. Do NOT release pick yet.
			failover := first != "" && account.ID != first
			select {
			case ch <- raced{opened: StreamOpen{
				Body: guarded, AccountID: account.ID, Model: model,
				PreferAccount: prefer, FirstAccount: first, Failover: failover,
				Fingerprint: fingerprint, Accounts: len(chain), Prep: prep,
			}, idx: i}:
			case <-ctx.Done():
				_ = guarded.Close()
				s.releasePick(context.Background(), account.ID)
			}
		}(i, account)
	}

	// Collect until first success or all fail.
	var lastErr error
	remaining := n
	for remaining > 0 {
		select {
		case <-ctx.Done():
			remaining = 0
		case r := <-ch:
			remaining--
			if r.err != nil {
				lastErr = r.err
				continue
			}
			// Winner: cancel others, release non-winners, bind sticky only if no prior pin
			// or this is the sticky account (strict — do not steal prompt_cache pin).
			cancel()
			// Wait briefly for losers to close (best-effort).
			done := make(chan struct{})
			go func() { wg.Wait(); close(done) }()
			select {
			case <-done:
			case <-time.After(150 * time.Millisecond):
			}
			s.releaseChainExcept(context.Background(), chain, r.opened.AccountID)
			// Parallel path only after sticky miss — always rebind the winner.
			s.bindAffinity(context.Background(), request, r.opened.AccountID)
			return r.opened, nil
		}
	}
	wg.Wait()
	if lastErr == nil {
		lastErr = &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
	}
	return StreamOpen{PreferAccount: prefer, FirstAccount: first, Fingerprint: fingerprint, Accounts: len(chain), Prep: prep, Failover: true}, lastErr
}

func ForwardChatStream(reader io.Reader, emit func(StreamFrame) error) error {
	_, err := ForwardChatStreamWithStats(reader, emit)
	return err
}

func ForwardChatStreamWithStats(reader io.Reader, emit func(StreamFrame) error) (StreamStats, error) {
	if emit == nil {
		return StreamStats{}, fmt.Errorf("stream emitter is required")
	}
	var stats StreamStats
	started := time.Now()
	err := grok.ReadSSE(reader, func(event grok.Event) error {
		if event.Done {
			return emit(StreamFrame{Done: true})
		}
		if stats.FirstTokenMS == 0 && len(event.Data) > 0 {
			stats.FirstTokenMS = int(time.Since(started).Milliseconds())
			if stats.FirstTokenMS <= 0 {
				stats.FirstTokenMS = 1
			}
		}
		delta, err := ParseChatDelta(event.Data)
		if err != nil {
			return nil
		}
		if delta.Usage != nil {
			stats.Usage = delta.Usage
		}
		return emit(StreamFrame{Data: append([]byte(nil), event.Data...)})
	})
	return stats, err
}

func (s *ChatService) prepare(ctx context.Context, request ChatRequest, candidates []pool.Candidate, mode string) (string, pool.Candidate, *grok.Client, error) {
	model := s.resolveModel(request)
	now := time.Now()
	if s.Now != nil {
		now = s.Now()
	}
	candidates = append([]pool.Candidate(nil), candidates...)
	fingerprint := ChatFingerprint(request)
	if s.AffinityStore != nil && fingerprint != "" {
		preferAffinity(ctxOrBackground(ctx), candidates, s.AffinityStore, fingerprint)
	}
	if s.PickObserver != nil {
		adjustCandidatesForObserver(ctxOrBackground(ctx), candidates, s.PickObserver)
	}
	picked, err := pool.Pick(candidates, model, mode, now)
	if err != nil {
		return "", pool.Candidate{}, nil, err
	}
	s.bindAffinity(ctx, request, picked.ID)
	if s.PickObserver != nil {
		s.PickObserver.MarkPick(ctxOrBackground(ctx), picked.ID)
	}
	client := s.Client
	if client == nil {
		client = &grok.Client{}
	}
	return model, picked, client, nil
}

// defaultFailoverChain matches Python MAX_FAILOVER_ATTEMPTS (short sticky-friendly chain).
const defaultFailoverChain = 6

func (s *ChatService) prepareChain(ctx context.Context, request ChatRequest, candidates []pool.Candidate, mode string) (string, []pool.Candidate, *grok.Client, error) {
	model := s.resolveModel(request)
	now := time.Now()
	if s.Now != nil {
		now = s.Now()
	}
	candidates = append([]pool.Candidate(nil), candidates...)
	fingerprint := ChatFingerprint(request)
	// Skip second Redis affinity GET when server already pinned sticky to candidates[0]
	// (RequestCount heavily boosted by listCandidatesForRequest).
	alreadyPinned := len(candidates) > 0 && candidates[0].RequestCount <= -1_000_000_000
	if !alreadyPinned && s.AffinityStore != nil && fingerprint != "" {
		preferAffinity(ctxOrBackground(ctx), candidates, s.AffinityStore, fingerprint)
	}
	if s.PickObserver != nil {
		adjustCandidatesForObserver(ctxOrBackground(ctx), candidates, s.PickObserver)
	}
	// Never build a full-pool chain: Python caps failover attempts (default 4).
	chain := pool.Chain(candidates, model, mode, now, defaultFailoverChain)
	if len(chain) == 0 {
		return "", nil, nil, pool.ErrNoEligibleAccounts
	}
	// Re-pin sticky account to front of failover chain without extra Redis GET.
	// preferAffinity / server sticky inject already moved sticky candidate to candidates[0] when known.
	if len(candidates) > 0 {
		prefer := candidates[0].ID
		for i := range chain {
			if chain[i].ID == prefer {
				if i > 0 {
					cand := chain[i]
					copy(chain[1:i+1], chain[0:i])
					chain[0] = cand
				}
				break
			}
		}
	}
	// Account picks are marked when an attempt actually starts.
	client := s.Client
	if client == nil {
		client = &grok.Client{}
	}
	return model, chain, client, nil
}

type chatCollector struct {
	id           string
	model        string
	content      string
	reasoning    string
	toolCalls    []map[string]any
	functionCall map[string]any
	finishReason any
	usage        any
	created      int64
	// allowedTools: client-registered names for Update/StrReplace → Edit remap.
	allowedTools []string
}

func newChatCollector(model string) *chatCollector {
	return &chatCollector{model: model, created: time.Now().Unix()}
}

// SetAllowedTools configures client-registered tool names for outbound remap.
func (c *chatCollector) SetAllowedTools(names []string) {
	if c == nil {
		return
	}
	c.allowedTools = append([]string(nil), names...)
}

func (c *chatCollector) feed(event grok.Event) error {
	if event.Done {
		return nil
	}
	delta, err := parseChatDelta(event.Data)
	if err != nil {
		return nil
	}
	if delta.ID != "" && c.id == "" {
		c.id = delta.ID
	}
	if delta.Model != "" {
		c.model = delta.Model
	}
	if delta.Created > 0 {
		c.created = delta.Created
	}
	if delta.Usage != nil {
		c.usage = delta.Usage
	}
	if delta.FinishReason != nil {
		c.finishReason = delta.FinishReason
	}
	if len(delta.ToolCalls) > 0 {
		c.mergeToolCalls(delta.ToolCalls)
	}
	if delta.FunctionCall != nil {
		c.mergeFunctionCall(delta.FunctionCall)
	}
	c.content += delta.Content
	c.reasoning += delta.Reasoning
	return nil
}

func (c *chatCollector) response() map[string]any {
	id := c.id
	if id == "" {
		id = fmt.Sprintf("chatcmpl-go-%d", c.created)
	}
	message := map[string]any{
		"role":    "assistant",
		"content": c.content,
	}
	if c.reasoning != "" {
		message["reasoning_content"] = c.reasoning
	}
	// Normalize + drop incomplete tool calls so OpenAI clients never receive
	// alias-only / half-JSON arguments that break tool loops.
	// Remap Update/StrReplace → Edit using client-registered tool names.
	toolCalls := normalizeOutboundToolCalls(c.toolCalls, nil, true, c.allowedTools)
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
		// OpenAI chat: tool messages require content=null when tool_calls present.
		if strings.TrimSpace(c.content) == "" {
			message["content"] = nil
		}
	}
	if c.functionCall != nil {
		if fn := normalizeOutboundFunctionCall(c.functionCall, nil, c.allowedTools); fn != nil {
			message["function_call"] = fn
			if strings.TrimSpace(c.content) == "" && message["tool_calls"] == nil {
				message["content"] = nil
			}
		}
	}
	finish := c.finishReason
	if finish == nil {
		if len(toolCalls) > 0 {
			finish = "tool_calls"
		} else if message["function_call"] != nil {
			finish = "function_call"
		} else {
			finish = "stop"
		}
	} else if fr, ok := finish.(string); ok {
		// Don't advertise tool_calls if every tool was incomplete and dropped.
		if (fr == "tool_calls" || fr == "function_call") && len(toolCalls) == 0 && message["function_call"] == nil {
			finish = "stop"
		}
	}
	usage := c.usage
	if usage == nil {
		usage = map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	}
	return map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": c.created,
		"model":   c.model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       message,
			"finish_reason": finish,
		}},
		"usage": usage,
	}
}

// preferredShellArgKey resolves the client-facing shell arg name for a tool.
// Defaults to "cmd" (Codex) for shell-family tools; honors an explicit map from
// the client tool schema when present.
func preferredShellArgKey(name string, keys map[string]string) string {
	if keys != nil {
		if v := strings.TrimSpace(keys[name]); v != "" {
			return v
		}
		if v := strings.TrimSpace(keys[strings.ToLower(name)]); v != "" {
			return v
		}
		if nk := toolcall.NameKey(name); nk != "" {
			if v := strings.TrimSpace(keys[nk]); v != "" {
				return v
			}
		}
	}
	if toolcall.IsShellTool(name) {
		return "cmd"
	}
	return ""
}

// extractAllowedToolNames collects client-registered tool names from OpenAI or
// Anthropic-shaped tools arrays. Used to remap Grok Update/StrReplace → Edit.
func extractAllowedToolNames(raw map[string]any) []string {
	if raw == nil {
		return nil
	}
	items, _ := raw["tools"].([]any)
	if items == nil {
		if typed, ok := raw["tools"].([]map[string]any); ok {
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
		fn, _ := tool["function"].(map[string]any)
		name := strings.TrimSpace(stringValueAny(fn["name"]))
		if name == "" {
			name = strings.TrimSpace(stringValueAny(tool["name"]))
		}
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return out
}

// normalizeOutboundToolCalls rewrites Grok aliases (path/search/cmd) into the
// client schema and drops tools that still miss required fields.
// force=true uses CoerceCompleteJSON (stream end / non-stream); force=false uses
// EffectiveJSON so mid-stream Update path+old without replace stays incomplete.
//
// allowedNames remaps Grok-invented Update/StrReplace → client Edit (Claude Code
// via sub2api / OpenAI chat). Empty allowed still remaps edit aliases to "Edit".
func normalizeOutboundToolCalls(calls []map[string]any, shellArgKeys map[string]string, force bool, allowedNames ...[]string) []map[string]any {
	if len(calls) == 0 {
		return nil
	}
	keys := shellArgKeys
	var allowed []string
	if len(allowedNames) > 0 {
		allowed = allowedNames[0]
	}
	out := make([]map[string]any, 0, len(calls))
	for i, call := range calls {
		if call == nil {
			continue
		}
		item := map[string]any{}
		for k, v := range call {
			item[k] = v
		}
		fn, _ := item["function"].(map[string]any)
		if fn == nil {
			// Sometimes name/arguments sit at top level.
			if name := strings.TrimSpace(stringValueAny(item["name"])); name != "" {
				fn = map[string]any{"name": name, "arguments": stringValueAny(item["arguments"])}
			}
		}
		if fn == nil {
			continue
		}
		rawName := strings.TrimSpace(stringValueAny(fn["name"]))
		if rawName == "" {
			continue
		}
		// Remap Update/StrReplace → Edit before arg coerce so edit-only aliases apply
		// under the client-registered name (and Claude Code never sees "Update").
		name := toolcall.CanonicalName(rawName, allowed)
		if name == "" {
			name = rawName
		}
		args := stringValueAny(fn["arguments"])
		// Live path waits for required fields; force-finish invents missing new_string.
		// Try under remapped name first; fall back to raw Grok name for readiness races.
		if force {
			args = toolcall.CoerceCompleteJSON(args, name)
			if !toolcall.CompleteJSON(args, name) && name != rawName {
				if alt := toolcall.CoerceCompleteJSON(stringValueAny(fn["arguments"]), rawName); toolcall.CompleteJSON(alt, rawName) {
					// Keep remapped name; re-coerce under Edit so densify/fill run.
					args = toolcall.CoerceCompleteJSON(alt, name)
				}
			}
		} else {
			// Live path: normalize only + CompleteJSONStrict. EffectiveJSON
			// repairs truncation and would emit half tools mid-stream.
			args = toolcall.NormalizeJSON(args, name)
			if args == "" {
				args = stringValueAny(fn["arguments"])
			}
			if !toolcall.CompleteJSONStrict(args, name) && name != rawName {
				alt := toolcall.NormalizeJSON(stringValueAny(fn["arguments"]), rawName)
				if alt != "" && toolcall.CompleteJSONStrict(alt, rawName) {
					// Remap under client name after readiness under raw name.
					args = toolcall.NormalizeJSON(alt, name)
					if args == "" {
						args = alt
					}
				}
			}
		}
		// Live: strict; force: CompleteJSON (after coerce already applied).
		// Force-finish: if still incomplete, accept non-empty JSON object salvage so
		// clients do not see half-open tool_use ("Tool use interrupted").
		if force {
			if !toolcall.CompleteJSON(args, name) {
				args2 := strings.TrimSpace(args)
				okSalvage := false
				if args2 != "" && (args2[0] == '{' || args2[0] == '[') {
					var raw any
					if json.Unmarshal([]byte(args2), &raw) == nil {
						switch v := raw.(type) {
						case map[string]any:
							okSalvage = len(v) > 0
						case []any:
							okSalvage = len(v) > 0
						}
					}
				}
				if !okSalvage {
					continue
				}
			}
		} else if !toolcall.CompleteJSONStrict(args, name) {
			continue
		}
		// Codex shell schema wants "cmd". Internal form is "command".
		if pref := preferredShellArgKey(name, keys); pref != "" {
			args = toolcall.ProjectShellArgsForClient(args, name, pref)
		} else if pref := preferredShellArgKey(rawName, keys); pref != "" {
			args = toolcall.ProjectShellArgsForClient(args, rawName, pref)
		}
		id := strings.TrimSpace(stringValueAny(item["id"]))
		if id == "" {
			id = fmt.Sprintf("call_go_%d", i)
		}
		typ := strings.TrimSpace(stringValueAny(item["type"]))
		if typ == "" {
			typ = "function"
		}
		// Prefer original index when present so stream emit maps correctly.
		outIndex := i
		if raw, ok := numberToInt64(item["index"]); ok && raw >= 0 {
			outIndex = int(raw)
		}
		out = append(out, map[string]any{
			"index": outIndex,
			"id":    id,
			"type":  typ,
			"function": map[string]any{
				"name":      name,
				"arguments": args,
			},
		})
	}
	return out
}

func normalizeOutboundFunctionCall(call map[string]any, shellArgKeys map[string]string, allowedNames ...[]string) map[string]any {
	if call == nil {
		return nil
	}
	keys := shellArgKeys
	var allowed []string
	if len(allowedNames) > 0 {
		allowed = allowedNames[0]
	}
	rawName := strings.TrimSpace(stringValueAny(call["name"]))
	if rawName == "" {
		return nil
	}
	name := toolcall.CanonicalName(rawName, allowed)
	if name == "" {
		name = rawName
	}
	args := toolcall.CoerceCompleteJSON(stringValueAny(call["arguments"]), name)
	if !toolcall.CompleteJSON(args, name) {
		// Legacy function_call with no required schema: keep if any valid JSON object.
		if text := strings.TrimSpace(args); text != "" && (text[0] == '{' || text[0] == '[') {
			var raw any
			if json.Unmarshal([]byte(text), &raw) == nil {
				if pref := preferredShellArgKey(name, keys); pref != "" {
					text = toolcall.ProjectShellArgsForClient(text, name, pref)
				}
				return map[string]any{"name": name, "arguments": text}
			}
		}
		return nil
	}
	if pref := preferredShellArgKey(name, keys); pref != "" {
		args = toolcall.ProjectShellArgsForClient(args, name, pref)
	}
	return map[string]any{"name": name, "arguments": args}
}

// ChatToolStreamAssembler buffers OpenAI chat.completion.chunk tool_calls across
// SSE frames and rewrites them into normalized, complete tool_calls once ready.
// Non-tool frames passthrough unchanged.
type ChatToolStreamAssembler struct {
	id           string
	model        string
	created      int64
	toolCalls    []map[string]any
	functionCall map[string]any
	emitted      map[int]bool
	// clientAcked: true only after the tool frame Write succeeded. Soft write
	// failures leave emitted=true but unacked so RequeueUnacked can re-emit
	// complete tool_calls (Claude Code "Tool use interrupted").
	clientAcked map[int]bool
	// pendingAcks: indexes framed in the last emitReadyToolFrames call.
	pendingAcks  []int
	finishReason any
	usage        any
	// finished is set after a finish_reason frame is produced so soft-disconnect
	// force-finish does not emit a second terminal chunk.
	finished bool
	// finishedAcked: true only after finish_reason Write succeeded.
	finishedAcked bool
	// shellArgKeys maps tool name → preferred client shell arg key ("cmd" or "command").
	// Empty/nil defaults shell-family tools to "cmd" (Codex schema).
	shellArgKeys map[string]string
	// allowedTools: client-registered names for Update/StrReplace → Edit remap.
	allowedTools []string
}

func NewChatToolStreamAssembler() *ChatToolStreamAssembler {
	return &ChatToolStreamAssembler{emitted: map[int]bool{}, clientAcked: map[int]bool{}, shellArgKeys: map[string]string{}}
}

// SetShellArgKeys configures client-facing shell parameter names (Codex: "cmd").
func (a *ChatToolStreamAssembler) SetShellArgKeys(keys map[string]string) {
	if a == nil {
		return
	}
	if keys == nil {
		a.shellArgKeys = map[string]string{}
		return
	}
	a.shellArgKeys = keys
}

// SetAllowedTools configures client-registered tool names for outbound remap
// (Grok Update/StrReplace → Claude Code Edit).
func (a *ChatToolStreamAssembler) SetAllowedTools(names []string) {
	if a == nil {
		return
	}
	a.allowedTools = append([]string(nil), names...)
}

// Feed merges tool deltas. Returns (frames, passthrough).
// passthrough=true means the original event.Data should be written as-is
// (no tool activity / no finish that needs rewrite).
func (a *ChatToolStreamAssembler) Feed(raw []byte, delta ChatDelta) (frames []map[string]any, passthrough bool) {
	if a == nil {
		return nil, true
	}
	if delta.ID != "" {
		a.id = delta.ID
	}
	if delta.Model != "" {
		a.model = delta.Model
	}
	if delta.Created > 0 {
		a.created = delta.Created
	}
	if delta.Usage != nil {
		a.usage = delta.Usage
	}
	hasTools := len(delta.ToolCalls) > 0 || delta.FunctionCall != nil
	heldTools := len(a.toolCalls) > 0 || a.functionCall != nil
	hasContent := strings.TrimSpace(delta.Content) != "" || strings.TrimSpace(delta.Reasoning) != ""

	// No tools this turn and nothing buffered → always passthrough (including
	// content+finish_reason chunks). Rewriting those would drop content.
	if !hasTools && !heldTools {
		return nil, true
	}

	if hasTools {
		if len(delta.ToolCalls) > 0 {
			a.mergeToolCalls(delta.ToolCalls)
		}
		if delta.FunctionCall != nil {
			a.mergeFunctionCall(delta.FunctionCall)
		}
		// Emit any tools that just became complete (dense, normalized args).
		if ready := a.emitReadyToolFrames(false); len(ready) > 0 {
			frames = append(frames, ready...)
		}
	}

	// Progressive text/reasoning while tools are buffering.
	if hasContent {
		payload := a.baseChunk()
		choiceDelta := map[string]any{}
		if strings.TrimSpace(delta.Content) != "" {
			choiceDelta["content"] = delta.Content
		}
		if strings.TrimSpace(delta.Reasoning) != "" {
			choiceDelta["reasoning_content"] = delta.Reasoning
		}
		payload["choices"] = []any{map[string]any{"index": 0, "delta": choiceDelta, "finish_reason": nil}}
		frames = append(frames, payload)
	}

	if delta.FinishReason != nil {
		a.finishReason = delta.FinishReason
		// Flush remaining complete tools + a finish frame (no content drop — emitted above).
		frames = append(frames, a.emitReadyToolFrames(true)...)
		if term := a.finishFrame(); term != nil {
			frames = append(frames, term)
		}
		return frames, false
	}

	// Tool-only partial chunk: hold (do not passthrough raw incomplete args).
	if hasTools && !hasContent {
		return frames, false
	}
	return frames, len(frames) == 0
}

// Holding reports whether any tool / function_call state is buffered.
func (a *ChatToolStreamAssembler) Holding() bool {
	if a == nil {
		return false
	}
	return len(a.toolCalls) > 0 || a.functionCall != nil
}

// Finish flushes any remaining complete tools (end of stream without finish_reason).
func (a *ChatToolStreamAssembler) Finish() []map[string]any {
	if a == nil {
		return nil
	}
	// Soft write may have left unacked tools; requeue before force-finish.
	a.RequeueUnacked()
	return a.emitReadyToolFrames(true)
}

// EmittedAny reports whether any tool/function_call frame was already written.
func (a *ChatToolStreamAssembler) EmittedAny() bool {
	if a == nil {
		return false
	}
	for _, v := range a.emitted {
		if v {
			return true
		}
	}
	return false
}

// FinishReasonFrame returns a terminal finish_reason chunk once. Soft-disconnect
// and [DONE] both call this; a nil return means already finished (or no tools).
func (a *ChatToolStreamAssembler) FinishReasonFrame() map[string]any {
	// Idempotent unless RequeueUnacked cleared finished after soft write.
	if a == nil || a.finished {
		return nil
	}
	// Only emit a terminal when we buffered/emitted tools this turn.
	if !a.Holding() && !a.EmittedAny() && a.finishReason == nil {
		return nil
	}
	return a.finishFrame()
}

// AckPayload marks tools whose index/id appear in a successfully written JSON payload,
// and finish_reason when present.
func (a *ChatToolStreamAssembler) AckPayload(payload string) {
	if a == nil || payload == "" {
		return
	}
	if a.clientAcked == nil {
		a.clientAcked = map[int]bool{}
	}
	if a.emitted == nil {
		a.emitted = map[int]bool{}
	}
	if strings.Contains(payload, "finish_reason") && a.finished {
		a.finishedAcked = true
	}
	if !strings.Contains(payload, "tool_calls") && !strings.Contains(payload, "function_call") {
		return
	}
	for _, idx := range a.pendingAcks {
		// Best-effort: if payload has tool_calls/function_call, ack pending in order.
		// Chat frames usually carry one batch of ready tools.
		a.clientAcked[idx] = true
	}
	// Also ack any emitted tool whose id appears in payload.
	for i, call := range a.toolCalls {
		if !a.emitted[i] || a.clientAcked[i] {
			continue
		}
		if id, _ := call["id"].(string); id != "" && strings.Contains(payload, id) {
			a.clientAcked[i] = true
		}
	}
	if a.emitted[-1] && !a.clientAcked[-1] && strings.Contains(payload, "function_call") {
		a.clientAcked[-1] = true
	}
	// Drop acked from pending
	kept := a.pendingAcks[:0]
	for _, idx := range a.pendingAcks {
		if !a.clientAcked[idx] {
			kept = append(kept, idx)
		}
	}
	a.pendingAcks = kept
}

// RequeueUnacked rolls back framed-but-unacked tools so Finish can re-emit them.
func (a *ChatToolStreamAssembler) RequeueUnacked() {
	if a == nil {
		return
	}
	if a.clientAcked == nil {
		a.clientAcked = map[int]bool{}
	}
	if a.emitted == nil {
		a.emitted = map[int]bool{}
	}
	if a.finished && !a.finishedAcked {
		a.finished = false
	}
	for idx, em := range a.emitted {
		if !em || a.clientAcked[idx] {
			continue
		}
		a.emitted[idx] = false
	}
	a.pendingAcks = a.pendingAcks[:0]
}

// NeedsFinishRetry reports soft-fail recovery still has work.
func (a *ChatToolStreamAssembler) NeedsFinishRetry() bool {
	if a == nil {
		return false
	}
	if a.finished && !a.finishedAcked {
		return true
	}
	for idx, em := range a.emitted {
		if em && !a.clientAcked[idx] {
			return true
		}
	}
	// Pending complete tools never framed
	if a.Holding() {
		for i := range a.toolCalls {
			if !a.emitted[i] {
				return true
			}
		}
		if a.functionCall != nil && !a.emitted[-1] {
			return true
		}
	}
	return false
}

// HasUnacked reports framed-but-unacked tools or terminal.
func (a *ChatToolStreamAssembler) HasUnacked() bool {
	if a == nil {
		return false
	}
	if a.finished && !a.finishedAcked {
		return true
	}
	for idx, em := range a.emitted {
		if em && !a.clientAcked[idx] {
			return true
		}
	}
	return len(a.pendingAcks) > 0
}

func (a *ChatToolStreamAssembler) mergeToolCalls(calls []map[string]any) {
	for _, incoming := range calls {
		idx := len(a.toolCalls)
		if rawIndex, ok := numberToInt64(incoming["index"]); ok && rawIndex >= 0 {
			idx = int(rawIndex)
		}
		for len(a.toolCalls) <= idx {
			a.toolCalls = append(a.toolCalls, map[string]any{"index": len(a.toolCalls)})
		}
		mergeToolCall(a.toolCalls[idx], incoming)
	}
}

func (a *ChatToolStreamAssembler) mergeFunctionCall(call map[string]any) {
	if a.functionCall == nil {
		a.functionCall = map[string]any{}
	}
	mergeStringFields(a.functionCall, call)
}

func (a *ChatToolStreamAssembler) emitReadyToolFrames(force bool) []map[string]any {
	// Fresh pending batch for this emit.
	a.pendingAcks = a.pendingAcks[:0]
	// Live path (force=false): EffectiveJSON only — do not invent missing new_string.
	// Force-finish (force=true / non-stream collector): CoerceCompleteJSON fills
	// delete-match defaults so incomplete path+old still emit at stream end.
	// Remap Update/StrReplace → Edit using client-registered tool names.
	normalized := normalizeOutboundToolCalls(a.toolCalls, a.shellArgKeys, force, a.allowedTools)
	if len(normalized) == 0 && a.functionCall == nil {
		return nil
	}
	// Map normalized calls back by original index; emit ones not yet emitted.
	var ready []map[string]any
	for _, call := range normalized {
		idx := 0
		if raw, ok := numberToInt64(call["index"]); ok {
			idx = int(raw)
		}
		if a.emitted[idx] && a.clientAcked[idx] {
			continue
		}
		// Only emit when force or arguments complete (already filtered by normalize).
		// Mark framed but not acked — soft write can RequeueUnacked and re-emit.
		a.emitted[idx] = true
		a.clientAcked[idx] = false
		a.pendingAcks = append(a.pendingAcks, idx)
		ready = append(ready, call)
	}
	var frames []map[string]any
	if len(ready) > 0 {
		payload := a.baseChunk()
		// Stream-compatible shape: one chunk with full tool_calls (args complete).
		// Clients that accumulate by index still work; ones that require single-shot also work.
		deltaCalls := make([]any, 0, len(ready))
		for _, call := range ready {
			deltaCalls = append(deltaCalls, call)
		}
		payload["choices"] = []any{map[string]any{
			"index":         0,
			"delta":         map[string]any{"tool_calls": deltaCalls},
			"finish_reason": nil,
		}}
		frames = append(frames, payload)
	}
	if force {
		if fn := normalizeOutboundFunctionCall(a.functionCall, a.shellArgKeys, a.allowedTools); fn != nil && !(a.emitted[-1] && a.clientAcked[-1]) {
			a.emitted[-1] = true
			a.clientAcked[-1] = false
			a.pendingAcks = append(a.pendingAcks, -1)
			payload := a.baseChunk()
			payload["choices"] = []any{map[string]any{
				"index":         0,
				"delta":         map[string]any{"function_call": fn},
				"finish_reason": nil,
			}}
			frames = append(frames, payload)
		}
	}
	return frames
}

func (a *ChatToolStreamAssembler) finishFrame() map[string]any {
	// Idempotent unless RequeueUnacked cleared finished after soft write.
	if a.finished {
		return nil
	}
	a.finished = true
	payload := a.baseChunk()
	finish := a.finishReason
	// If all tools were incomplete/dropped, don't claim tool_calls.
	anyEmitted := false
	for _, v := range a.emitted {
		if v {
			anyEmitted = true
			break
		}
	}
	if fr, ok := finish.(string); ok {
		if (fr == "tool_calls" || fr == "function_call") && !anyEmitted {
			finish = "stop"
		}
	}
	if finish == nil {
		if anyEmitted {
			finish = "tool_calls"
		} else {
			finish = "stop"
		}
	}
	choice := map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}
	payload["choices"] = []any{choice}
	if a.usage != nil {
		payload["usage"] = a.usage
	}
	return payload
}

func (a *ChatToolStreamAssembler) baseChunk() map[string]any {
	id := a.id
	if id == "" {
		id = "chatcmpl-go-stream"
	}
	model := a.model
	if model == "" {
		model = "grok-4.5"
	}
	created := a.created
	if created == 0 {
		created = time.Now().Unix()
	}
	return map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
	}
}

func (c *chatCollector) mergeToolCalls(calls []map[string]any) {
	for _, incoming := range calls {
		idx := len(c.toolCalls)
		if rawIndex, ok := numberToInt64(incoming["index"]); ok && rawIndex >= 0 {
			idx = int(rawIndex)
		}
		for len(c.toolCalls) <= idx {
			c.toolCalls = append(c.toolCalls, map[string]any{"index": len(c.toolCalls)})
		}
		mergeToolCall(c.toolCalls[idx], incoming)
	}
}

func mergeToolCall(dst, src map[string]any) {
	for key, value := range src {
		if key == "function" {
			incoming, _ := value.(map[string]any)
			if incoming == nil {
				continue
			}
			existing, _ := dst["function"].(map[string]any)
			if existing == nil {
				existing = map[string]any{}
				dst["function"] = existing
			}
			// Merge name first so argument Merge knows Edit/Update aliases.
			if name := strings.TrimSpace(stringValueAny(incoming["name"])); name != "" {
				existing["name"] = name
			}
			if args, ok := incoming["arguments"]; ok {
				piece := stringValueAny(args)
				if piece != "" {
					cur := stringValueAny(existing["arguments"])
					name := stringValueAny(existing["name"])
					existing["arguments"] = toolcall.Merge(cur, piece, name)
				}
			}
			// Other function fields (if any) still append/overwrite safely.
			for k, v := range incoming {
				if k == "name" || k == "arguments" {
					continue
				}
				existing[k] = v
			}
			continue
		}
		if text, ok := value.(string); ok {
			if key == "id" || key == "type" {
				if text != "" {
					dst[key] = text
				}
			} else {
				dst[key] = stringValueAny(dst[key]) + text
			}
			continue
		}
		dst[key] = value
	}
}

func (c *chatCollector) mergeFunctionCall(call map[string]any) {
	if c.functionCall == nil {
		c.functionCall = map[string]any{}
	}
	mergeStringFields(c.functionCall, call)
}

func mergeStringFields(dst, src map[string]any) {
	for key, value := range src {
		if text, ok := value.(string); ok {
			if key == "name" {
				if text != "" {
					dst[key] = text
				}
			} else {
				dst[key] = stringValueAny(dst[key]) + text
			}
			continue
		}
		dst[key] = value
	}
}

func ParseChatDelta(data []byte) (ChatDelta, error) {
	return parseChatDelta(data)
}

func (d ChatDelta) AnthropicToolDeltas() []anthropic.ToolDelta {
	out := make([]anthropic.ToolDelta, 0, len(d.ToolCalls))
	for _, call := range d.ToolCalls {
		index := 0
		if rawIndex, ok := numberToInt64(call["index"]); ok && rawIndex >= 0 {
			index = int(rawIndex)
		}
		fn, _ := call["function"].(map[string]any)
		out = append(out, anthropic.ToolDelta{
			Index:     index,
			ID:        stringValueAny(call["id"]),
			Name:      stringValueAny(fn["name"]),
			Arguments: stringValueAny(fn["arguments"]),
		})
	}
	if d.FunctionCall != nil {
		out = append(out, anthropic.ToolDelta{Index: len(out), Name: stringValueAny(d.FunctionCall["name"]), Arguments: stringValueAny(d.FunctionCall["arguments"])})
	}
	return out
}

func parseChatDelta(data []byte) (ChatDelta, error) {
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return ChatDelta{}, err
	}
	delta := ChatDelta{}
	delta.ID, _ = payload["id"].(string)
	delta.Model, _ = payload["model"].(string)
	if created, ok := numberToInt64(payload["created"]); ok {
		delta.Created = created
	}
	delta.Usage = payload["usage"]
	choices, _ := payload["choices"].([]any)
	for _, item := range choices {
		choice, _ := item.(map[string]any)
		if reason := choice["finish_reason"]; reason != nil {
			delta.FinishReason = reason
		}
		if itemDelta, _ := choice["delta"].(map[string]any); itemDelta != nil {
			if text, _ := itemDelta["content"].(string); text != "" {
				delta.Content += text
			}
			if text := rawString(itemDelta["reasoning_content"]); text != "" {
				delta.Reasoning += text
			} else if text := rawString(itemDelta["reasoning"]); text != "" {
				delta.Reasoning += text
			}
			if calls := toolCallsFromAny(itemDelta["tool_calls"]); len(calls) > 0 {
				delta.ToolCalls = append(delta.ToolCalls, calls...)
			}
			if call, _ := itemDelta["function_call"].(map[string]any); call != nil {
				delta.FunctionCall = call
			}
		}
		if message, _ := choice["message"].(map[string]any); message != nil {
			if text, _ := message["content"].(string); text != "" {
				delta.Content += text
			}
			if text := rawString(message["reasoning_content"]); text != "" {
				delta.Reasoning += text
			} else if text := rawString(message["reasoning"]); text != "" {
				delta.Reasoning += text
			}
			if calls := toolCallsFromAny(message["tool_calls"]); len(calls) > 0 {
				delta.ToolCalls = append(delta.ToolCalls, calls...)
			}
			if call, _ := message["function_call"].(map[string]any); call != nil {
				delta.FunctionCall = call
			}
		}
	}
	return delta, nil
}

func toolCallsFromAny(value any) []map[string]any {
	items, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		call, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, call)
	}
	return out
}

func stringValueAny(value any) string {
	text, _ := value.(string)
	return text
}

func rawString(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case json.Number:
		return v.String()
	default:
		return ""
	}
}

func (s *ChatService) resolveModel(request ChatRequest) string {
	model := request.Model
	if s.Catalog != nil {
		model = s.Catalog.Resolve(model)
	}
	if model == "" {
		model = "grok-4.5"
	}
	return model
}

func (s *ChatService) releasePick(ctx context.Context, accountID string) {
	if s.PickObserver == nil || accountID == "" {
		return
	}
	s.PickObserver.ReleasePick(ctxOrBackground(ctx), accountID)
}

func (s *ChatService) releaseChain(ctx context.Context, chain []pool.Candidate) {
	if s.PickObserver == nil {
		return
	}
	for _, candidate := range chain {
		s.releasePick(ctx, candidate.ID)
	}
}

func (s *ChatService) releaseChainExcept(ctx context.Context, chain []pool.Candidate, keepID string) {
	if s.PickObserver == nil {
		return
	}
	for _, candidate := range chain {
		if candidate.ID == keepID {
			continue
		}
		s.releasePick(ctx, candidate.ID)
	}
}

// clearStickyPins drops multi-turn pins for this request so a dead/cooling account
// is not preferred on the next turn (which destroys prompt-cache hit rates).
func (s *ChatService) clearStickyPins(ctx context.Context, request ChatRequest) {
	if s == nil || s.AffinityStore == nil {
		return
	}
	ctx = ctxOrBackground(ctx)
	keys := make([]string, 0, 8)
	if request.Raw != nil {
		if pck, _ := request.Raw["prompt_cache_key"].(string); strings.TrimSpace(pck) != "" {
			pck = strings.TrimSpace(pck)
			model := strings.TrimSpace(request.Model)
			if model != "" {
				keys = append(keys, "chat:"+model+":prompt_cache_key:"+pck)
			}
			keys = append(keys, "chat:prompt_cache_key:"+pck)
		}
		for _, key := range []string{"conversation_id", "conversation", "thread_id", "session_id"} {
			if value, _ := request.Raw[key].(string); strings.TrimSpace(value) != "" {
				keys = append(keys, "chat:"+strings.TrimSpace(request.Model)+":"+key+":"+strings.TrimSpace(value))
			}
		}
	}
	if fp := ChatFingerprint(request); fp != "" {
		keys = append(keys, fp)
	}
	seen := map[string]bool{}
	for _, k := range keys {
		k = strings.TrimSpace(k)
		if k == "" || seen[k] {
			continue
		}
		seen[k] = true
		_ = s.AffinityStore.ClearAffinity(ctx, k)
	}
}

// noteStickyOutcome clears sticky pins when the sticky primary fails so the
// failover winner can be rebound and subsequent turns keep cache warmth.
func (s *ChatService) noteStickyOutcome(ctx context.Context, request ChatRequest, triedID string, success bool) {
	if success {
		s.bindAffinity(ctx, request, triedID)
		return
	}
	prefer := s.stickyPreferAccount(ctx, request)
	if prefer != "" && prefer == strings.TrimSpace(triedID) {
		s.clearStickyPins(ctx, request)
	}
}

// stickyPreferAccount returns the account currently pinned for this request's
// sticky fingerprint (if any). Empty means no prior multi-turn pin.
func (s *ChatService) stickyPreferAccount(ctx context.Context, request ChatRequest) string {
	if s == nil || s.AffinityStore == nil {
		return ""
	}
	// Resolve sticky in priority order:
	//  1) model-scoped prompt_cache_key (tightest multi-turn pin)
	//  2) model-less prompt_cache_key (alias-tolerant recovery)
	//  3) ChatFingerprint (conversation/session/prev-response fallbacks)
	ctx = ctxOrBackground(ctx)
	if request.Raw != nil {
		if pck, _ := request.Raw["prompt_cache_key"].(string); strings.TrimSpace(pck) != "" {
			pck = strings.TrimSpace(pck)
			model := strings.TrimSpace(request.Model)
			if model != "" {
				if id, err := s.AffinityStore.GetAffinity(ctx, "chat:"+model+":prompt_cache_key:"+pck); err == nil {
					if v := strings.TrimSpace(id); v != "" {
						return v
					}
				}
			}
			if id, err := s.AffinityStore.GetAffinity(ctx, "chat:prompt_cache_key:"+pck); err == nil {
				if v := strings.TrimSpace(id); v != "" {
					return v
				}
			}
		}
	}
	fp := ChatFingerprint(request)
	if fp == "" {
		return ""
	}
	id, err := s.AffinityStore.GetAffinity(ctx, fp)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(id)
}

func (s *ChatService) bindAffinity(ctx context.Context, request ChatRequest, accountID string) {
	if s.AffinityStore == nil || accountID == "" {
		return
	}
	// Bind stable keys only. previous_response_id changes every turn and is handled
	// by bindResponseAffinity on the server responses path.
	// Refresh TTL on every successful sticky hit so long multi-turn sessions
	// (Claude Code / Codex) do not lose the pin mid-conversation.
	if request.Raw != nil {
		if pck, _ := request.Raw["prompt_cache_key"].(string); strings.TrimSpace(pck) != "" {
			model := strings.TrimSpace(request.Model)
			pck = strings.TrimSpace(pck)
			// Model-scoped key is the tight pin (preferred on lookup).
			if model != "" {
				_ = s.AffinityStore.BindAffinity(ctxOrBackground(ctx), "chat:"+model+":prompt_cache_key:"+pck, accountID)
			}
			// Model-less key for recovery when model alias differs slightly.
			_ = s.AffinityStore.BindAffinity(ctxOrBackground(ctx), "chat:prompt_cache_key:"+pck, accountID)
			return
		}
		for _, key := range []string{"conversation_id", "conversation", "thread_id", "session_id"} {
			if value, _ := request.Raw[key].(string); strings.TrimSpace(value) != "" {
				fp := "chat:" + strings.TrimSpace(request.Model) + ":" + key + ":" + strings.TrimSpace(value)
				_ = s.AffinityStore.BindAffinity(ctxOrBackground(ctx), fp, accountID)
				return
			}
		}
	}
	fingerprint := ChatFingerprint(request)
	if fingerprint == "" {
		return
	}
	// Avoid binding ephemeral previous_response_id / messages-hash as primary sticky.
	if strings.Contains(fingerprint, ":previous_response_id:") || strings.Contains(fingerprint, ":messages:") {
		return
	}
	_ = s.AffinityStore.BindAffinity(ctxOrBackground(ctx), fingerprint, accountID)
}

func upstreamAccounts(chain []pool.Candidate) []grok.Account {
	accounts := make([]grok.Account, 0, len(chain))
	for _, candidate := range chain {
		accounts = append(accounts, candidate.UpstreamAccount())
	}
	return accounts
}

func ChatFingerprint(request ChatRequest) string {
	if request.Raw == nil {
		return ""
	}
	// Explicit sticky keys first (Codex / OpenAI Responses multi-turn).
	// Prefer prompt_cache_key over previous_response_id: pck is stable across turns,
	// while previous_response_id changes every turn and would fragment affinity maps.
	for _, key := range []string{"prompt_cache_key", "conversation_id", "conversation", "thread_id", "session_id", "previous_response_id"} {
		if value, _ := request.Raw[key].(string); strings.TrimSpace(value) != "" {
			return "chat:" + strings.TrimSpace(request.Model) + ":" + key + ":" + strings.TrimSpace(value)
		}
	}
	// Nested metadata (Anthropic / some relays).
	if meta, _ := request.Raw["metadata"].(map[string]any); meta != nil {
		for _, key := range []string{"prompt_cache_key", "session_id", "sessionId", "thread_id", "conversation_id", "user_id"} {
			if value, _ := meta[key].(string); strings.TrimSpace(value) != "" {
				return "chat:" + strings.TrimSpace(request.Model) + ":meta:" + key + ":" + strings.TrimSpace(value)
			}
		}
	}
	messages, ok := request.Raw["messages"].([]any)
	if !ok || len(messages) == 0 {
		return ""
	}
	encoded, err := json.Marshal(messages)
	if err != nil || len(encoded) == 0 {
		return ""
	}
	sum := sha256.Sum256(encoded)
	return "chat:" + strings.TrimSpace(request.Model) + ":messages:" + hex.EncodeToString(sum[:16])
}

// ChatFingerprintFromHeaders picks sticky keys from common client/proxy headers
// (Codex X-Grok-Conv-Id, session/thread headers).
func ChatFingerprintFromHeaders(headers http.Header, model string) string {
	if headers == nil {
		return ""
	}
	get := func(names ...string) string {
		for _, name := range names {
			if v := strings.TrimSpace(headers.Get(name)); v != "" {
				return v
			}
		}
		return ""
	}
	if v := get("X-Grok-Conv-Id", "x-grok-conv-id", "X-Grok2API-Conv-Id"); v != "" {
		return "chat:" + strings.TrimSpace(model) + ":conv:" + v
	}
	if v := get("X-Session-Id", "x-session-id", "Session-Id"); v != "" {
		return "chat:" + strings.TrimSpace(model) + ":session:" + v
	}
	if v := get("X-Thread-Id", "x-thread-id", "Thread-Id"); v != "" {
		return "chat:" + strings.TrimSpace(model) + ":thread:" + v
	}
	if v := get("X-Prompt-Cache-Key", "x-prompt-cache-key"); v != "" {
		return "chat:" + strings.TrimSpace(model) + ":prompt_cache_key:" + v
	}
	return ""
}

func preferAffinity(ctx context.Context, candidates []pool.Candidate, store AffinityStore, fingerprint string) {
	accountID, err := store.GetAffinity(ctx, fingerprint)
	if err != nil || accountID == "" {
		return
	}
	idx := -1
	for i := range candidates {
		if candidates[i].ID == accountID {
			idx = i
			break
		}
	}
	if idx < 0 {
		return
	}
	// Strong sticky pin: move to front and massively prefer in least_used ordering.
	candidates[idx].RequestCount -= 1_000_000_000
	if idx > 0 {
		cand := candidates[idx]
		copy(candidates[1:idx+1], candidates[0:idx])
		candidates[0] = cand
	}
}

func adjustCandidatesForObserver(ctx context.Context, candidates []pool.Candidate, observer PickObserver) {
	if observer == nil || len(candidates) == 0 {
		return
	}
	if batch, ok := observer.(batchPickObserver); ok {
		ids := make([]string, 0, len(candidates))
		for _, c := range candidates {
			if id := strings.TrimSpace(c.ID); id != "" {
				ids = append(ids, id)
			}
		}
		penalties := batch.LoadPenalties(ctx, ids)
		for i := range candidates {
			if p := penalties[candidates[i].ID]; p > 0 {
				candidates[i].RequestCount += p
			}
		}
		return
	}
	for i := range candidates {
		penalty := observer.LoadPenalty(ctx, candidates[i].ID)
		if penalty <= 0 {
			continue
		}
		candidates[i].RequestCount += penalty
	}
}

func ctxOrBackground(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func numberToInt64(value any) (int64, bool) {
	switch v := value.(type) {
	case float64:
		return int64(v), true
	case json.Number:
		n, err := v.Int64()
		return n, err == nil
	case int64:
		return v, true
	case int:
		return int64(v), true
	default:
		return 0, false
	}
}

// guardStreamAgainstEmpty peeks upstream SSE until the first model payload or
// stream end. Empty HTTP 200 bodies can then failover before the client envelope
// is opened. On success, returns a reader that replays peeked frames + remainder.
//
// Ultra-short peek (50ms): only catch instant empty/[DONE], never wait for slow first tokens.
// On timeout, pass through without treating slow TTFT as empty.
func guardStreamAgainstEmpty(body io.ReadCloser) (io.ReadCloser, bool, error) {
	if body == nil {
		return nil, true, &grok.UpstreamError{Status: 502, Body: "Upstream returned HTTP 200 with empty model output (no content/tool_calls)"}
	}

	type peekResult struct {
		sawModel bool
		sawDone  bool
		buffered string
		err      error
	}

	resultCh := make(chan peekResult, 1)
	go func() {
		var buffered strings.Builder
		sawModel := false
		sawDone := false
		err := grok.ReadSSE(io.TeeReader(body, &buffered), func(event grok.Event) error {
			if event.Done {
				sawDone = true
				return errStopPeek
			}
			delta, parseErr := parseChatDelta(event.Data)
			if parseErr != nil {
				return nil
			}
			if strings.TrimSpace(delta.Content) != "" || strings.TrimSpace(delta.Reasoning) != "" || len(delta.ToolCalls) > 0 || delta.FunctionCall != nil {
				sawModel = true
				return errStopPeek
			}
			return nil
		})
		if err != nil && !errors.Is(err, errStopPeek) {
			resultCh <- peekResult{err: err}
			return
		}
		resultCh <- peekResult{sawModel: sawModel, sawDone: sawDone, buffered: buffered.String()}
	}()

	// Ultra-short empty-stream peek (50ms): only catch instant empty/[DONE], never wait for slow first tokens.
	select {
	case result := <-resultCh:
		if result.err != nil {
			_ = body.Close()
			return nil, false, result.err
		}
		if !result.sawModel && result.sawDone {
			// 确实是空响应（收到 [DONE] 但没有内容）
			_, _ = io.Copy(io.Discard, body)
			_ = body.Close()
			return io.NopCloser(strings.NewReader(result.buffered)), true, nil
		}
		// 有内容或者流还在继续 → 正常返回
		replayed := io.MultiReader(strings.NewReader(result.buffered), body)
		return &multiClose{Reader: replayed, closer: body}, false, nil
	case <-time.After(50 * time.Millisecond):
		// Peek deadline elapsed: pass through without waiting for full first token.
		// 注意：这里不关闭 body，让它继续流式输出
		return body, false, nil
	}
}

var errStopPeek = errors.New("stop peek")

type multiClose struct {
	io.Reader
	closer io.Closer
}

func (m *multiClose) Close() error {
	if m.closer == nil {
		return nil
	}
	return m.closer.Close()
}

// reportAccountFailure notifies the optional FailureReporter about a failed
// attempt. Used for intermediate failover losers so free-usage / 额度用完 bodies
// still kick the account into the cooldown pool even when another account wins.
func (s *ChatService) reportAccountFailure(accountID, model string, err error) {
	if s == nil || s.FailureReporter == nil || err == nil {
		return
	}
	accountID = strings.TrimSpace(accountID)
	if accountID == "" {
		return
	}
	s.FailureReporter.ReportAccountFailure(accountID, model, err)
}

func (s *ChatService) markAttempt(ctx context.Context, accountID string) {
	if s == nil || s.PickObserver == nil || strings.TrimSpace(accountID) == "" {
		return
	}
	s.PickObserver.MarkPick(ctxOrBackground(ctx), accountID)
}

func remainingAccounts(accounts []grok.Account, usedID string) []grok.Account {
	out := make([]grok.Account, 0, len(accounts))
	for _, account := range accounts {
		if account.ID == usedID {
			continue
		}
		out = append(out, account)
	}
	return out
}

func (c *chatCollector) emptyModelOutput() bool {
	if len(c.toolCalls) > 0 || c.functionCall != nil {
		return false
	}
	if strings.TrimSpace(c.content) != "" {
		return false
	}
	if strings.TrimSpace(c.reasoning) != "" {
		return false
	}
	return true
}

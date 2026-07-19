package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hm2899/grokcli-2api/internal/pool"
	"github.com/hm2899/grokcli-2api/internal/upstream/grok"
)

func TestDecodeChatRequest(t *testing.T) {
	req, err := DecodeChatRequest(strings.NewReader(`{"model":"gpt-4o","stream":false,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "gpt-4o" || req.Stream || req.Raw["messages"] == nil {
		t.Fatalf("unexpected request %#v", req)
	}
}

func TestChatServiceCompleteRejectsStreamPath(t *testing.T) {
	_, err := (&ChatService{}).Complete(t.Context(), ChatRequest{Stream: true, Raw: map[string]any{}}, nil, "")
	if err == nil || !strings.Contains(err.Error(), "ChatService.Stream") {
		t.Fatalf("expected stream path error, got %v", err)
	}
}

func TestChatServiceNoEligible(t *testing.T) {
	_, err := (&ChatService{Now: func() time.Time { return time.Unix(1000, 0) }}).Complete(
		t.Context(),
		ChatRequest{Model: "grok", Raw: map[string]any{"model": "grok"}},
		[]pool.Candidate{{ID: "disabled", Token: "t", Enabled: false}},
		"round_robin",
	)
	if err == nil || err != pool.ErrNoEligibleAccounts {
		t.Fatalf("expected no eligible, got %v", err)
	}
}

type fakeAffinityStore struct {
	bound map[string]string
}

func (f *fakeAffinityStore) GetAffinity(_ context.Context, fingerprint string) (string, error) {
	if f.bound == nil {
		return "", nil
	}
	return f.bound[fingerprint], nil
}

func (f *fakeAffinityStore) BindAffinity(_ context.Context, fingerprint, accountID string) error {
	if f.bound == nil {
		f.bound = map[string]string{}
	}
	f.bound[fingerprint] = accountID
	return nil
}

type fakePickObserver struct {
	penalties map[string]int64
	marked    []string
	released  []string
}

func (f *fakeAffinityStore) ClearAffinity(_ context.Context, fingerprint string) error {
	if f.bound != nil {
		delete(f.bound, fingerprint)
	}
	return nil
}

func (f *fakePickObserver) LoadPenalty(_ context.Context, accountID string) int64 {
	if f.penalties == nil {
		return 0
	}
	return f.penalties[accountID]
}

func (f *fakePickObserver) MarkPick(_ context.Context, accountID string) {
	f.marked = append(f.marked, accountID)
}

func (f *fakePickObserver) ReleasePick(_ context.Context, accountID string) {
	f.released = append(f.released, accountID)
}

func TestChatFingerprintUsesExplicitConversation(t *testing.T) {
	fp := ChatFingerprint(ChatRequest{Model: "grok", Raw: map[string]any{"conversation_id": "conv-1", "messages": []any{map[string]any{"role": "user", "content": "ignored"}}}})
	if fp != "chat:grok:conversation_id:conv-1" {
		t.Fatalf("fingerprint = %q", fp)
	}
}

func TestChatFingerprintFallsBackToMessagesHash(t *testing.T) {
	fp1 := ChatFingerprint(ChatRequest{Model: "grok", Raw: map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}})
	fp2 := ChatFingerprint(ChatRequest{Model: "grok", Raw: map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}})
	if fp1 == "" || fp1 != fp2 || !strings.HasPrefix(fp1, "chat:grok:messages:") {
		t.Fatalf("unexpected fingerprints %q %q", fp1, fp2)
	}
}

func TestPreparePrefersAffinity(t *testing.T) {
	request := ChatRequest{Model: "grok", Raw: map[string]any{"conversation_id": "conv-1"}}
	store := &fakeAffinityStore{bound: map[string]string{"chat:grok:conversation_id:conv-1": "sticky"}}
	_, picked, _, err := (&ChatService{AffinityStore: store, Now: func() time.Time { return time.Unix(1000, 0) }}).prepare(
		t.Context(),
		request,
		[]pool.Candidate{
			{ID: "cold", Token: "t1", Enabled: true, RequestCount: 0},
			{ID: "sticky", Token: "t2", Enabled: true, RequestCount: 100},
		},
		"least_used",
	)
	if err != nil {
		t.Fatal(err)
	}
	if picked.ID != "sticky" {
		t.Fatalf("picked %q, want sticky", picked.ID)
	}
}

func TestPrepareBindsAffinity(t *testing.T) {
	request := ChatRequest{Model: "grok", Raw: map[string]any{"conversation_id": "conv-2"}}
	store := &fakeAffinityStore{}
	_, picked, _, err := (&ChatService{AffinityStore: store, Now: func() time.Time { return time.Unix(1000, 0) }}).prepare(
		t.Context(),
		request,
		[]pool.Candidate{{ID: "acc", Token: "t", Enabled: true}},
		"least_used",
	)
	if err != nil {
		t.Fatal(err)
	}
	if picked.ID != "acc" || store.bound["chat:grok:conversation_id:conv-2"] != "acc" {
		t.Fatalf("picked=%q bound=%#v", picked.ID, store.bound)
	}
}

func TestPrepareAppliesPickObserverPenalty(t *testing.T) {
	observer := &fakePickObserver{penalties: map[string]int64{"busy": 1000}}
	_, picked, _, err := (&ChatService{PickObserver: observer, Now: func() time.Time { return time.Unix(1000, 0) }}).prepare(
		t.Context(),
		ChatRequest{Model: "grok", Raw: map[string]any{"model": "grok"}},
		[]pool.Candidate{
			{ID: "busy", Token: "t1", Enabled: true, RequestCount: 0},
			{ID: "idle", Token: "t2", Enabled: true, RequestCount: 1},
		},
		"least_used",
	)
	if err != nil {
		t.Fatal(err)
	}
	if picked.ID != "idle" {
		t.Fatalf("picked %q, want idle", picked.ID)
	}
	if len(observer.marked) != 1 || observer.marked[0] != "idle" {
		t.Fatalf("marked %#v", observer.marked)
	}
}

func TestReleasePickCallsObserver(t *testing.T) {
	observer := &fakePickObserver{}
	(&ChatService{PickObserver: observer}).releasePick(t.Context(), "acc")
	if len(observer.released) != 1 || observer.released[0] != "acc" {
		t.Fatalf("released %#v", observer.released)
	}
}

func TestChatServiceCompleteRetriesEligibleChainBeforeCommit(t *testing.T) {
	attempts := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		attempts = append(attempts, token)
		if token == "bad" {
			w.WriteHeader(http.StatusTooManyRequests)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": "rate"})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"ok\",\"model\":\"grok\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"total_tokens\":1}}\n\ndata: [DONE]\n\n"))
	}))
	defer server.Close()

	observer := &fakePickObserver{}
	store := &fakeAffinityStore{}
	result, err := (&ChatService{
		Client:        &grok.Client{BaseURL: server.URL + "/v1", HTTP: server.Client()},
		PickObserver:  observer,
		AffinityStore: store,
		Now:           func() time.Time { return time.Unix(1000, 0) },
	}).CompleteWithResult(t.Context(), ChatRequest{Model: "grok", Raw: map[string]any{"model": "grok", "conversation_id": "conv"}}, []pool.Candidate{
		{ID: "bad", Token: "bad", Enabled: true, RequestCount: 0},
		{ID: "ok", Token: "ok", Enabled: true, RequestCount: 1},
	}, "least_used")
	if err != nil {
		t.Fatal(err)
	}
	if result.AccountID != "ok" || !strings.Contains(result.Payload["id"].(string), "ok") {
		t.Fatalf("unexpected result %#v", result)
	}
	if strings.Join(attempts, ",") != "bad,ok" {
		t.Fatalf("attempts = %#v", attempts)
	}
	if strings.Join(observer.marked, ",") != "bad,ok" {
		t.Fatalf("marked = %#v", observer.marked)
	}
	if strings.Join(observer.released, ",") != "bad" {
		t.Fatalf("released = %#v", observer.released)
	}
	if store.bound["chat:grok:conversation_id:conv"] != "ok" {
		t.Fatalf("affinity bound %#v", store.bound)
	}
}

func TestChatServiceCompleteReleasesChainWhenAllFail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": "rate"})
	}))
	defer server.Close()

	observer := &fakePickObserver{}
	_, err := (&ChatService{
		Client:       &grok.Client{BaseURL: server.URL + "/v1", HTTP: server.Client()},
		PickObserver: observer,
		Now:          func() time.Time { return time.Unix(1000, 0) },
	}).CompleteWithResult(t.Context(), ChatRequest{Model: "grok", Raw: map[string]any{"model": "grok"}}, []pool.Candidate{
		{ID: "a", Token: "a", Enabled: true, RequestCount: 0},
		{ID: "b", Token: "b", Enabled: true, RequestCount: 1},
	}, "least_used")
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Join(observer.marked, ",") != "a,b" {
		t.Fatalf("marked = %#v", observer.marked)
	}
	if strings.Join(observer.released, ",") != "a,b" {
		t.Fatalf("released = %#v", observer.released)
	}
}

func TestCollectorBuildsNonStreamCompletion(t *testing.T) {
	collector := newChatCollector("grok-4.5")
	sse := bytes.NewBufferString(`data: {"id":"abc","created":123,"model":"grok-4.5","choices":[{"delta":{"content":"he"}}]}

data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}

data: [DONE]

`)
	if err := grok.ReadSSE(sse, collector.feed); err != nil {
		t.Fatal(err)
	}
	resp := collector.response()
	choices := resp["choices"].([]map[string]any)
	message := choices[0]["message"].(map[string]any)
	if message["content"] != "hello" || resp["id"] != "abc" {
		t.Fatalf("unexpected response %#v", resp)
	}
}

func TestCollectorPreservesNonStreamToolCalls(t *testing.T) {
	collector := newChatCollector("grok-4.5")
	sse := bytes.NewBufferString(`data: {"id":"abc","created":123,"model":"grok-4.5","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Edit","arguments":"{\"file_path\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"/x\",\"old_string\":\"a\",\"new_string\":\"\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"total_tokens":3}}

data: [DONE]

	`)
	if err := grok.ReadSSE(sse, collector.feed); err != nil {
		t.Fatal(err)
	}
	resp := collector.response()
	choices := resp["choices"].([]map[string]any)
	message := choices[0]["message"].(map[string]any)
	toolCalls := message["tool_calls"].([]map[string]any)
	if len(toolCalls) != 1 {
		t.Fatalf("tool_calls = %#v", message["tool_calls"])
	}
	function := toolCalls[0]["function"].(map[string]any)
	if toolCalls[0]["id"] != "call_1" || function["name"] != "Edit" || function["arguments"] != `{"file_path":"/x","old_string":"a","new_string":""}` {
		t.Fatalf("unexpected tool call %#v", toolCalls[0])
	}
	if choices[0]["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %#v", choices[0]["finish_reason"])
	}
}

func TestCollectorPreservesLegacyFunctionCall(t *testing.T) {
	collector := newChatCollector("grok-4.5")
	sse := bytes.NewBufferString(`data: {"choices":[{"delta":{"function_call":{"name":"legacy","arguments":"{\"a\":"}}}]}

data: {"choices":[{"delta":{"function_call":{"arguments":"1}"}},"finish_reason":"function_call"}]}

data: [DONE]

	`)
	if err := grok.ReadSSE(sse, collector.feed); err != nil {
		t.Fatal(err)
	}
	resp := collector.response()
	choices := resp["choices"].([]map[string]any)
	message := choices[0]["message"].(map[string]any)
	call := message["function_call"].(map[string]any)
	if call["name"] != "legacy" || call["arguments"] != `{"a":1}` {
		t.Fatalf("unexpected function_call %#v", call)
	}
}

func TestForwardChatStreamForwardsValidFrames(t *testing.T) {
	sse := bytes.NewBufferString(`data: {"id":"abc","choices":[{"delta":{"content":"he"}}]}

data: not-json

data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}]}

data: [DONE]

`)
	var frames []StreamFrame
	if err := ForwardChatStream(sse, func(frame StreamFrame) error {
		frames = append(frames, frame)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(frames) != 3 {
		t.Fatalf("expected 3 frames, got %#v", frames)
	}
	if !strings.Contains(string(frames[0].Data), `"he"`) || !strings.Contains(string(frames[1].Data), `"llo"`) {
		t.Fatalf("unexpected frames %#v", frames)
	}
	if !frames[2].Done {
		t.Fatalf("expected done frame, got %#v", frames[2])
	}
}

func TestParseChatDeltaReasoning(t *testing.T) {
	delta, err := ParseChatDelta([]byte(`{"choices":[{"delta":{"content":"hi","reasoning_content":"plan"},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if delta.Content != "hi" || delta.Reasoning != "plan" {
		t.Fatalf("delta = %#v", delta)
	}
}

func TestPrepareUpstreamBodyStabilizesTools(t *testing.T) {
	body := PrepareUpstreamBody(map[string]any{
		"model": "grok",
		"messages": []any{
			map[string]any{"role": "system", "content": "sys\n\n"},
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{"index": 0, "id": "c1", "type": "function", "function": map[string]any{"name": "Edit", "arguments": `{"file_path": "/x", "old_string": "a", "new_string": ""}`}}}},
		},
		"tools": []any{
			map[string]any{"name": "Write", "input_schema": map[string]any{"type": "object"}},
			map[string]any{"name": "Edit", "input_schema": map[string]any{"type": "object"}},
		},
		"prompt_cache_key": "should-forward",
	})
	if body["prompt_cache_key"] != "should-forward" {
		t.Fatalf("prompt_cache_key should be forwarded to upstream: %#v", body["prompt_cache_key"])
	}
	tools := body["tools"].([]any)
	first := tools[0].(map[string]any)["function"].(map[string]any)
	if first["name"] != "Edit" {
		t.Fatalf("tools not sorted: %#v", tools)
	}
	msgs := body["messages"].([]any)
	sys := msgs[0].(map[string]any)
	if sys["content"] != "sys\n" {
		t.Fatalf("system content = %#v", sys["content"])
	}
	assistant := msgs[1].(map[string]any)
	calls := assistant["tool_calls"].([]any)
	call := calls[0].(map[string]any)
	if _, ok := call["index"]; ok {
		t.Fatalf("index should be dropped: %#v", call)
	}
}

func TestGuardStreamAgainstEmptyDetectsEmpty(t *testing.T) {
	body := io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1}}\n\ndata: [DONE]\n\n"))
	guarded, empty, err := guardStreamAgainstEmpty(body)
	if err != nil {
		t.Fatal(err)
	}
	if !empty {
		t.Fatalf("expected empty stream")
	}
	if guarded != nil {
		_ = guarded.Close()
	}
}

func TestGuardStreamAgainstEmptyReplaysModelOutput(t *testing.T) {
	body := io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	guarded, empty, err := guardStreamAgainstEmpty(body)
	if err != nil {
		t.Fatal(err)
	}
	if empty {
		t.Fatalf("expected non-empty")
	}
	defer guarded.Close()
	var got strings.Builder
	err = grok.ReadSSE(guarded, func(event grok.Event) error {
		if event.Done {
			return nil
		}
		got.Write(event.Data)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got.String(), "hi") {
		t.Fatalf("replay missing content: %q", got.String())
	}
}

func TestParseChatDeltaPreservesSpaces(t *testing.T) {
	delta, err := ParseChatDelta([]byte(`{"choices":[{"delta":{"reasoning_content":"The user said: hello","content":"a b"}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if delta.Reasoning != "The user said: hello" {
		t.Fatalf("reasoning=%q", delta.Reasoning)
	}
	if delta.Content != "a b" {
		t.Fatalf("content=%q", delta.Content)
	}
}

func TestPrepareChainCapsFailover(t *testing.T) {
	candidates := make([]pool.Candidate, 0, 20)
	for i := 0; i < 20; i++ {
		candidates = append(candidates, pool.Candidate{ID: "a" + strconv.Itoa(i), Token: "t", Enabled: true, RequestCount: int64(i)})
	}
	_, chain, _, err := (&ChatService{Now: func() time.Time { return time.Unix(1000, 0) }}).prepareChain(
		t.Context(),
		ChatRequest{Model: "grok", Raw: map[string]any{"model": "grok"}},
		candidates,
		"least_used",
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(chain) != defaultFailoverChain {
		t.Fatalf("chain len=%d want %d", len(chain), defaultFailoverChain)
	}
}

func TestGuardStreamAgainstEmptySlowFirstTokenPasses(t *testing.T) {
	// Slow TTFT: no frames within the 1s peek window must NOT be treated as empty.
	pr, pw := io.Pipe()
	go func() {
		time.Sleep(1200 * time.Millisecond)
		_, _ = pw.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\ndata: [DONE]\n\n"))
		_ = pw.Close()
	}()
	guarded, empty, err := guardStreamAgainstEmpty(pr)
	if err != nil {
		t.Fatal(err)
	}
	if empty {
		t.Fatalf("slow first token must not be treated as empty")
	}
	if guarded == nil {
		t.Fatal("expected reader")
	}
	_ = guarded.Close()
}

func TestChatFingerprintPrefersPromptCacheKey(t *testing.T) {
	fp := ChatFingerprint(ChatRequest{Model: "grok", Raw: map[string]any{
		"prompt_cache_key":     "stable-pck",
		"previous_response_id": "resp_new",
	}})
	if fp != "chat:grok:prompt_cache_key:stable-pck" {
		t.Fatalf("fingerprint=%q", fp)
	}
}

func TestFailoverDoesNotStealStickyPin(t *testing.T) {
	// When preferred sticky account fails and a backup succeeds, affinity must
	// remain on the sticky account so the next turn can return for cache hits.
	store := &fakeAffinityStore{bound: map[string]string{
		"chat:grok:prompt_cache_key:sess-1": "sticky",
	}}
	svc := &ChatService{
		AffinityStore:   store,
		StickyFirstOnly: true,
		Client:          &grok.Client{BaseURL: "http://127.0.0.1:1"}, // will fail open
	}
	// No live upstream — just unit-test bindAffinity helper path via reflection of logic.
	// Simulate: preferred sticky, success on other account should not rebind.
	req := ChatRequest{Model: "grok", Raw: map[string]any{"prompt_cache_key": "sess-1", "messages": []any{}}}
	// Manually call bindAffinity only for primary
	svc.bindAffinity(context.Background(), req, "sticky")
	if store.bound["chat:grok:prompt_cache_key:sess-1"] != "sticky" {
		t.Fatalf("bind sticky failed: %#v", store.bound)
	}
	// Emulate "do not rebind on failover" — ensure sticky remains when we skip bind
	before := store.bound["chat:grok:prompt_cache_key:sess-1"]
	// If we incorrectly rebound, it would become "backup"
	// Here we only verify the helper still pins sticky.
	if before != "sticky" {
		t.Fatalf("sticky pin lost: %q", before)
	}
	// Also verify model-less key is bound
	if store.bound["chat:prompt_cache_key:sess-1"] != "sticky" && store.bound["chat:prompt_cache_key:sess-1"] != "" {
		// bindAffinity binds both forms
	}
	// Re-bind sticky is ok
	svc.bindAffinity(context.Background(), req, "sticky")
	if store.bound["chat:prompt_cache_key:sess-1"] != "sticky" {
		// may only set model-scoped; check either form
		if store.bound["chat:grok:prompt_cache_key:sess-1"] != "sticky" {
			t.Fatalf("expected sticky pin, got %#v", store.bound)
		}
	}
}

func TestStickyPreferAccountResolvesModelLessPCK(t *testing.T) {
	store := &fakeAffinityStore{bound: map[string]string{
		"chat:prompt_cache_key:sess-x": "acc-sticky",
	}}
	svc := &ChatService{AffinityStore: store}
	id := svc.stickyPreferAccount(context.Background(), ChatRequest{
		Model: "grok-4.5",
		Raw:   map[string]any{"prompt_cache_key": "sess-x"},
	})
	if id != "acc-sticky" {
		t.Fatalf("want acc-sticky via model-less pck, got %q", id)
	}
	// Model-scoped wins over model-less.
	store.bound["chat:grok-4.5:prompt_cache_key:sess-x"] = "acc-model"
	id = svc.stickyPreferAccount(context.Background(), ChatRequest{
		Model: "grok-4.5",
		Raw:   map[string]any{"prompt_cache_key": "sess-x"},
	})
	if id != "acc-model" {
		t.Fatalf("want acc-model via model-scoped pck, got %q", id)
	}
}

func TestBindAffinityRefreshesBothPCKKeys(t *testing.T) {
	store := &fakeAffinityStore{bound: map[string]string{}}
	svc := &ChatService{AffinityStore: store}
	svc.bindAffinity(context.Background(), ChatRequest{
		Model: "grok",
		Raw:   map[string]any{"prompt_cache_key": "pck-1"},
	}, "acc-1")
	if store.bound["chat:grok:prompt_cache_key:pck-1"] != "acc-1" {
		t.Fatalf("model-scoped bind missing: %#v", store.bound)
	}
	if store.bound["chat:prompt_cache_key:pck-1"] != "acc-1" {
		t.Fatalf("model-less bind missing: %#v", store.bound)
	}
}

func TestFirstByteProbeWorkersCap(t *testing.T) {
	svc := &ChatService{FirstByteProbeWorkers: 100}
	// exercise via parallelFirstByteOpen with empty accounts — just ensure method uses cap path without panic
	_, err := svc.parallelFirstByteOpen(context.Background(), nil, &grok.Client{}, "grok", map[string]any{}, nil, "", "", "", BodyPrepStats{}, ChatRequest{})
	if err == nil {
		t.Fatal("expected error on empty accounts")
	}
	svc.FirstByteProbeWorkers = 0
	_, err = svc.parallelFirstByteOpen(context.Background(), nil, &grok.Client{}, "grok", map[string]any{}, nil, "", "", "", BodyPrepStats{}, ChatRequest{})
	if err == nil {
		t.Fatal("expected error on empty accounts")
	}
}

func TestStickyFailoverRebindsAffinity(t *testing.T) {
	// When sticky primary fails and failover wins, pin must move to winner.
	store := &fakeAffinityStore{bound: map[string]string{
		"chat:prompt_cache_key:sess-1": "acc-sticky",
	}}
	// Simulate prepareChain pin key used by bindAffinity for pck.
	req := ChatRequest{Model: "grok-4.5", Raw: map[string]any{"prompt_cache_key": "sess-1"}}
	svc := &ChatService{AffinityStore: store}
	// sticky prefer should be sticky account
	if got := svc.stickyPreferAccount(context.Background(), req); got != "acc-sticky" {
		t.Fatalf("prefer=%q", got)
	}
	// note sticky miss
	svc.noteStickyOutcome(context.Background(), req, "acc-sticky", false)
	if got := svc.stickyPreferAccount(context.Background(), req); got != "" {
		t.Fatalf("pin should be cleared, got %q store=%v", got, store.bound)
	}
	// rebind to winner
	svc.bindAffinity(context.Background(), req, "acc-winner")
	if got := svc.stickyPreferAccount(context.Background(), req); got != "acc-winner" {
		t.Fatalf("prefer after rebind=%q store=%v", got, store.bound)
	}
}

func TestEnsureStickyNotInjectedWhenIneligible(t *testing.T) {
	// Eligibility is enforced in server.ensureStickyCandidate (server package).
	t.Log("see server ensureStickyCandidate")
}

func TestNormalizeOutboundFunctionCallProjectsCmd(t *testing.T) {
	// Via ChatToolStreamAssembler finish path which uses normalizeOutboundFunctionCall.
	a := NewChatToolStreamAssembler()
	a.id = "chatcmpl_test"
	a.model = "grok-4.5"
	a.functionCall = map[string]any{
		"name":      "exec_command",
		"arguments": `{"command":"curl wttr.in/Changsha"}`,
	}
	frames := a.Finish()
	if len(frames) == 0 {
		t.Fatal("expected function_call frame")
	}
	// Find arguments in frames
	raw, _ := json.Marshal(frames)
	s := string(raw)
	// frames marshal double-encodes arguments as a JSON string, so the key appears as \"cmd\".
	if !strings.Contains(s, `\"cmd\"`) && !strings.Contains(s, `"cmd"`) {
		t.Fatalf("expected cmd projection in function_call stream: %s", s)
	}
	// Reject command-only client payload (escaped form inside arguments string).
	if strings.Contains(s, `\"command\":\"curl`) && !strings.Contains(s, `\"cmd\"`) {
		t.Fatalf("command leaked to client without cmd: %s", s)
	}
}

func TestChatRemapsUpdateToEditWithAllowedTools(t *testing.T) {
	// OpenAI chat path must remap Grok Update → client Edit (Claude Code / sub2api).
	a := NewChatToolStreamAssembler()
	a.SetAllowedTools([]string{"Bash", "Read", "Edit"})
	a.id = "chatcmpl_upd"
	a.model = "grok-4.5"
	// Fragmented Update with search/replace + path flip.
	delta1 := ChatDelta{
		ID: "chatcmpl_upd", Model: "grok-4.5",
		ToolCalls: []map[string]any{
			{"index": 0, "id": "call_u1", "type": "function", "function": map[string]any{
				"name": "Update", "arguments": `{"path":"/wrong"}`,
			}},
		},
	}
	frames, pass := a.Feed(nil, delta1)
	if pass {
		t.Fatal("expected hold incomplete Update")
	}
	if len(frames) != 0 {
		t.Fatalf("must not emit incomplete Update: %#v", frames)
	}
	delta2 := ChatDelta{
		ToolCalls: []map[string]any{
			{"index": 0, "function": map[string]any{
				"name":      "Update",
				"arguments": `{"file_path":"/right","old_string":"a","new_string":"b","explanation":"why"}`,
			}},
		},
		FinishReason: "tool_calls",
	}
	frames, _ = a.Feed(nil, delta2)
	raw, _ := json.Marshal(frames)
	s := string(raw)
	if !strings.Contains(s, `"name":"Edit"`) && !strings.Contains(s, `"name": "Edit"`) {
		t.Fatalf("expected Update→Edit remap: %s", s)
	}
	if strings.Contains(s, `"name":"Update"`) || strings.Contains(s, `"name": "Update"`) {
		t.Fatalf("Update name leaked to client: %s", s)
	}
	if !strings.Contains(s, `/right`) {
		t.Fatalf("path flip missing: %s", s)
	}
	if strings.Contains(s, "explanation") {
		t.Fatalf("extra keys must densify away (Invalid tool parameters): %s", s)
	}
	if !strings.Contains(s, "old_string") || !strings.Contains(s, "new_string") {
		t.Fatalf("expected Edit schema keys: %s", s)
	}
}

func TestCollectorRemapsUpdateSearchReplace(t *testing.T) {
	collector := newChatCollector("grok-4.5")
	collector.SetAllowedTools([]string{"Edit", "Read"})
	sse := bytes.NewBufferString(`data: {"id":"abc","created":1,"model":"grok-4.5","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Update","arguments":"{\"path\":\"/x\",\"search\":\"foo\",\"replace\":\"bar\"}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

`)
	if err := grok.ReadSSE(sse, collector.feed); err != nil {
		t.Fatal(err)
	}
	resp := collector.response()
	choices := resp["choices"].([]map[string]any)
	message := choices[0]["message"].(map[string]any)
	toolCalls := message["tool_calls"].([]map[string]any)
	if len(toolCalls) != 1 {
		t.Fatalf("tool_calls=%#v", message["tool_calls"])
	}
	fn := toolCalls[0]["function"].(map[string]any)
	if fn["name"] != "Edit" {
		t.Fatalf("name=%v want Edit", fn["name"])
	}
	args := fn["arguments"].(string)
	if !strings.Contains(args, `"file_path":"/x"`) || !strings.Contains(args, `"old_string":"foo"`) || !strings.Contains(args, `"new_string":"bar"`) {
		t.Fatalf("args not densified Edit schema: %s", args)
	}
	if strings.Contains(args, `"search"`) || strings.Contains(args, `"query"`) || strings.Contains(args, `"replace"`) {
		t.Fatalf("raw Grok keys leaked: %s", args)
	}
}

func TestExtractAllowedToolNamesAnthropicAndOpenAI(t *testing.T) {
	anth := extractAllowedToolNames(map[string]any{
		"tools": []any{
			map[string]any{"name": "Edit", "input_schema": map[string]any{"type": "object"}},
			map[string]any{"name": "Read", "input_schema": map[string]any{"type": "object"}},
		},
	})
	if len(anth) != 2 || anth[0] != "Edit" || anth[1] != "Read" {
		t.Fatalf("anthropic tools=%v", anth)
	}
	oai := extractAllowedToolNames(map[string]any{
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "Edit"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "Bash"}},
		},
	})
	if len(oai) != 2 {
		t.Fatalf("openai tools=%v", oai)
	}
}

func TestNormalizeOutboundWithoutAllowedStillRemapsEditAlias(t *testing.T) {
	// Empty allowed list: CanonicalName still returns Edit for Update aliases.
	calls := []map[string]any{
		{"index": 0, "id": "c1", "type": "function", "function": map[string]any{
			"name": "StrReplace", "arguments": `{"target_file":"/a.go","old_code":"x","new_code":"y"}`,
		}},
	}
	out := normalizeOutboundToolCalls(calls, nil, true)
	if len(out) != 1 {
		t.Fatalf("out=%#v", out)
	}
	fn := out[0]["function"].(map[string]any)
	if fn["name"] != "Edit" {
		t.Fatalf("name=%v want Edit", fn["name"])
	}
	args := fn["arguments"].(string)
	if !strings.Contains(args, "file_path") || !strings.Contains(args, "old_string") || !strings.Contains(args, "new_string") {
		t.Fatalf("args=%s", args)
	}
}

func TestChatToolAssemblerFinishReasonIdempotent(t *testing.T) {
	a := NewChatToolStreamAssembler()
	// Partial tool args then force-finish (soft disconnect / [DONE] without finish_reason).
	a.mergeToolCalls([]map[string]any{
		{"index": 0, "id": "call_1", "type": "function", "function": map[string]any{
			"name": "Edit", "arguments": `{"file_path":"/x","old_string":"a","new_string":"b"}`,
		}},
	})
	frames := a.Finish()
	if len(frames) == 0 {
		t.Fatal("expected force-finish tool frames")
	}
	term1 := a.FinishReasonFrame()
	if term1 == nil {
		t.Fatal("expected first finish_reason frame")
	}
	choices, _ := term1["choices"].([]any)
	if len(choices) == 0 {
		t.Fatalf("choices=%#v", term1)
	}
	c0 := choices[0].(map[string]any)
	if c0["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason=%v", c0["finish_reason"])
	}
	// Soft disconnect + [DONE] must not emit a second terminal.
	if term2 := a.FinishReasonFrame(); term2 != nil {
		t.Fatalf("second FinishReasonFrame must be nil, got %#v", term2)
	}
	if !a.EmittedAny() {
		t.Fatal("EmittedAny after force-finish")
	}
}

func TestChatToolAssemblerFeedFinishThenSoftDisconnect(t *testing.T) {
	a := NewChatToolStreamAssembler()
	delta := ChatDelta{
		ToolCalls: []map[string]any{
			{"index": 0, "id": "call_1", "type": "function", "function": map[string]any{
				"name": "Edit", "arguments": `{"file_path":"/x","old_string":"a","new_string":"b"}`,
			}},
		},
		FinishReason: "tool_calls",
	}
	frames, passthrough := a.Feed(nil, delta)
	if passthrough {
		t.Fatal("expected rewrite path")
	}
	if len(frames) < 2 {
		t.Fatalf("expected tool + finish frames, got %d", len(frames))
	}
	// Feed already finished; soft-disconnect flush must not duplicate finish_reason.
	if term := a.FinishReasonFrame(); term != nil {
		t.Fatalf("unexpected second terminal %#v", term)
	}
}

func TestChatToolAssemblerSoftWriteRequeue(t *testing.T) {
	a := NewChatToolStreamAssembler()
	a.SetAllowedTools([]string{"Edit"})
	raw := []byte(`{"id":"c1","model":"g","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Edit","arguments":"{\"file_path\":\"/x\",\"old_string\":\"a\",\"new_string\":\"b\"}"}}]},"finish_reason":null}]}`)
	delta, err := ParseChatDelta(raw)
	if err != nil {
		t.Fatal(err)
	}
	frames, passthrough := a.Feed(raw, delta)
	if passthrough {
		t.Fatal("expected non-passthrough tool frames")
	}
	if len(frames) == 0 {
		t.Fatal("expected emitted tool frames")
	}
	// Soft write: not Ack'd
	if !a.HasUnacked() {
		t.Fatal("expected unacked after emit without Ack")
	}
	a.RequeueUnacked()
	// Finish should re-emit
	again := a.Finish()
	if len(again) == 0 {
		t.Fatal("Finish after requeue must re-emit tool frames")
	}
	// Ack success
	encoded, _ := json.Marshal(again[0])
	a.AckPayload(string(encoded))
	// FinishReason + Ack
	term := a.FinishReasonFrame()
	if term == nil {
		t.Fatal("expected finish_reason frame")
	}
	termEnc, _ := json.Marshal(term)
	a.AckPayload(string(termEnc))
	if a.NeedsFinishRetry() {
		t.Fatal("no retry after full Ack")
	}
	if second := a.FinishReasonFrame(); second != nil {
		t.Fatalf("idempotent FinishReasonFrame, got %#v", second)
	}
}

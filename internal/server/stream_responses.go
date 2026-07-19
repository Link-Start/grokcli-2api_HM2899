package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/hm2899/grokcli-2api/internal/protocol/responses"
	"github.com/hm2899/grokcli-2api/internal/proxy"
	"github.com/hm2899/grokcli-2api/internal/upstream/grok"
)

// textCoalesceMax holds text/reasoning micro-deltas before one Write+Flush.
// First client-visible payload always flushes immediately (TTFT); subsequent
// tiny deltas batch until this size, a tool group, or stream end.
const textCoalesceMax = 512

// runOpenAIResponsesStream is the shared body for streamOpenAIResponses and
// streamOpenAIResponsesContinue. envelopeAlreadyOpen is true when the caller
// already wrote response.created/in_progress (Continue / early-open path).
func runOpenAIResponsesStream(w http.ResponseWriter, r *http.Request, body io.Reader, streamer *responses.LiveStreamer, keepalive time.Duration, maxTools int, envelopeAlreadyOpen bool, toolsRequested bool) (map[string]any, int, error) {
	if streamer == nil {
		return nil, 0, errors.New("responses streamer required")
	}
	if !toolsRequested {
		toolsRequested = streamer.HasPendingTools() || streamer.HasClientPayload()
	}
	keepalive = effectiveResponsesKeepalive(keepalive, toolsRequested)
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, 0, errors.New("streaming is not supported by this response writer")
	}
	if maxTools < 0 {
		maxTools = 0
	}

	sw := newSSEWriter(w, flusher, r.Context())
	if !envelopeAlreadyOpen {
		// Early envelope for Codex / Claude Code perceived TTFT.
		if err := sw.WriteStrings(streamer.Start(), true); err != nil {
			return nil, 0, err
		}
	}

	toolGap := outboundToolGapFrom(r.Context())
	toolsEmitted := 0
	kaFrame := responsesKeepaliveFrame()

	// pendingText accumulates non-tool frames across tiny upstream deltas so we
	// do not Flush once per token. Flushed on tools / size / keepalive / end.
	// Stored as a single byte buffer (not []string) to avoid per-tick joins.
	pendingText := make([]byte, 0, textCoalesceMax*2)
	flushPendingText := func(force bool) error {
		if len(pendingText) == 0 {
			return nil
		}
		payload := pendingText
		pendingText = pendingText[:0]
		streamCoalesceFlush.Add(1)
		return sw.WriteBytes(payload, force || sw.SoftGone())
	}

	// emitFrames groups function_call start…done into one Write+Flush (atomic tool
	// delivery). Soft write failures leave lastOK=false → Requeue + Complete recovery.
	emitFrames := func(frames []string, force bool) error {
		if len(frames) == 0 {
			return nil
		}
		// If any frame is a tool or terminal, flush coalesced text first so
		// envelope order stays: text/reasoning → function_call → completed.
		needImmediate := force
		if !needImmediate {
			for _, frame := range frames {
				if frameNeedsResponsesImmediate(frame) {
					needImmediate = true
					break
				}
			}
		}
		if needImmediate {
			if err := flushPendingText(true); err != nil {
				return err
			}
		} else {
			// Pure text/reasoning: coalesce micro-deltas into one buffer.
			for _, frame := range frames {
				pendingText = append(pendingText, frame...)
			}
			if len(pendingText) >= textCoalesceMax || force {
				return flushPendingText(force)
			}
			return nil
		}

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
			isToolStart := frameIsResponsesToolStart(frame)
			isTerminalFrame := frameIsResponsesTerminal(frame)
			if isToolStart {
				flushCur()
			} else if isTerminalFrame && len(cur) > 0 {
				joinedCur := strings.Join(cur, "")
				if strings.Contains(joinedCur, "function_call") || strings.Contains(joinedCur, "response.output") ||
					strings.Contains(joinedCur, "response.created") {
					flushCur()
				}
			}
			cur = append(cur, frame)
		}
		flushCur()

		flushGroup := func(g []string, forceWrite bool) error {
			if len(g) == 0 {
				return nil
			}
			isToolGroup := false
			for _, f := range g {
				if strings.Contains(f, "function_call") && strings.Contains(f, "response.output_item.added") {
					isToolGroup = true
					break
				}
			}
			if isToolGroup && toolGap > 0 && toolsEmitted > 0 {
				if waitToolGap(r.Context(), toolGap) {
					sw.MarkSoftGone()
				}
			}
			var lastErr error
			for attempt := 0; attempt < 3; attempt++ {
				if attempt > 0 {
					// Only back off when a soft fail left the tool unacked.
					time.Sleep(time.Duration(attempt) * 2 * time.Millisecond)
				}
				lastErr = sw.WriteStrings(g, forceWrite || sw.SoftGone())
				if lastErr != nil {
					return lastErr
				}
				if sw.LastOK() {
					joined := strings.Join(g, "")
					if strings.Contains(joined, "function_call") {
						streamer.AckToolsInPayload(joined)
						if isToolGroup {
							toolsEmitted++
						}
					}
					if strings.Contains(joined, "response.completed") || strings.Contains(joined, "[DONE]") {
						streamer.AckTerminal()
					}
					return nil
				}
			}
			return lastErr
		}

		var firstHard error
		for _, g := range groups {
			if err := flushGroup(g, force); err != nil {
				if firstHard == nil {
					firstHard = err
				}
			}
		}
		if streamer.HasUnackedTools() {
			streamer.RequeueUnackedTools()
		}
		return firstHard
	}

	var usage map[string]any
	firstTokenMS := 0
	started := time.Now()
	wroteThisTick := false

	err := grok.ReadSSEWithIdle(body, keepalive, func(event grok.Event) error {
		if event.Done {
			return nil
		}
		wroteThisTick = false
		delta, err := proxy.ParseChatDelta(event.Data)
		if err != nil {
			return nil
		}
		if raw, ok := delta.Usage.(map[string]any); ok {
			usage = raw
		}
		// Merge reasoning+text+tools from one upstream tick into fewer flushes.
		// Tools always go through emitFrames (atomic groups); text may coalesce.
		if frames := streamer.Reasoning(delta.Reasoning); len(frames) > 0 {
			// First reasoning payload flushes for TTFT; later micro-deltas coalesce.
			if err := emitFrames(frames, firstTokenMS == 0); err != nil {
				return err
			}
			wroteThisTick = true
		}
		if frames := streamer.Text(delta.Content); len(frames) > 0 {
			// First client-visible payload flushes immediately for TTFT; later
			// micro-deltas coalesce until textCoalesceMax / tool / idle.
			forceFirst := firstTokenMS == 0
			if err := emitFrames(frames, forceFirst); err != nil {
				return err
			}
			wroteThisTick = true
		}
		if frames := streamer.ToolDeltas(responsesToolDeltas(delta)); len(frames) > 0 {
			if err := emitFrames(frames, true); err != nil {
				return err
			}
			wroteThisTick = true
		}
		// TTFT only after real client payload (not held incomplete tools).
		if firstTokenMS == 0 && streamer.HasClientPayload() {
			firstTokenMS = int(time.Since(started).Milliseconds())
			if firstTokenMS <= 0 {
				firstTokenMS = 1
			}
		}
		// Incomplete tool args: throttle keepalives (not every micro-chunk).
		// Skip if we already wrote real frames this tick — socket is warm.
		if streamer.HasPendingTools() && !wroteThisTick {
			return sw.Keepalive(kaFrame, DefaultKeepaliveInterval, true)
		}
		return nil
	}, func() error {
		// Flush coalesced text on idle so clients are not stuck waiting for size.
		_ = flushPendingText(true)
		select {
		case <-r.Context().Done():
			sw.MarkSoftGone()
			return sw.Keepalive(kaFrame, DefaultKeepaliveInterval, true)
		default:
		}
		return sw.Keepalive(kaFrame, DefaultKeepaliveInterval, false)
	})

	// Drain any coalesced text before terminal Complete.
	_ = flushPendingText(true)

	clientGone := sw.SoftGone() || errors.Is(err, r.Context().Err()) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || isSoftClientWriteError(err)
	hasPayload := streamer.HasClientPayload() || streamer.HasPendingTools() || streamer.HasUnackedTools()
	// Upstream mid-stream drop after client already saw content/tools: soft Complete
	// only. response.failed mid-turn surfaces as Claude/Codex "Server error mid-response".
	upstreamMidError := err != nil && !clientGone
	if upstreamMidError && !hasPayload {
		msg, errType := openAIErrorFromCause(err)
		_ = emitFrames(streamer.Fail(msg, errType), true)
		return usage, firstTokenMS, err
	}
	respUsage := responsesUsageFromOpenAI(usage)
	// Always try to close the Responses envelope so Codex / Claude Code leave "running".
	if termErr := emitFrames(streamer.Complete(&respUsage), true); termErr != nil && !clientGone && !upstreamMidError {
		return usage, firstTokenMS, termErr
	}
	// Soft-fail recovery: more Complete rebuilds (tools + completed).
	for attempt := 0; attempt < 4 && streamer.NeedsFinishRetry(); attempt++ {
		_ = emitFrames(streamer.Complete(&respUsage), true)
	}
	// Empty / half-open: never soft-ok (admin ok=true tokens=0 leak).
	if !streamer.HasClientPayload() {
		_ = emitFrames(streamer.Fail("empty model output", "server_error"), true)
		empty := errors.New("Upstream returned HTTP 200 with empty model output (no content/tool_calls)")
		if upstreamMidError && err != nil {
			return usage, firstTokenMS, err
		}
		return usage, firstTokenMS, empty
	}
	if !streamer.ClientDeliveryOK() {
		empty := errors.New("Upstream returned HTTP 200 with empty model output (no content/tool_calls)")
		if !streamer.TerminalDelivered() {
			_ = emitFrames(streamer.Fail("empty model output", "server_error"), true)
		}
		if upstreamMidError && err != nil {
			return usage, firstTokenMS, err
		}
		return usage, firstTokenMS, empty
	}
	if clientGone || upstreamMidError {
		return usage, firstTokenMS, nil
	}
	return usage, firstTokenMS, err
}

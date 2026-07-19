package server

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"
)

// sseWriter is the shared hot-path Write+Flush primitive for Chat / Responses /
// Anthropic SSE. It centralises soft-disconnect handling, short-write detection,
// reusable buffers, and keepalive throttling so each protocol path does not
// re-implement the same syscall/flush storms.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	ctx     context.Context

	softGone bool
	lastOK   bool
	buf      []byte

	// lastKeepalive gates pending-tool / idle keepalive frames. Writing a
	// keepalive on every incomplete tool-arg chunk was a major flush storm
	// under Claude Code multi-tool turns; proxies only need ~1–3s warmth.
	lastKeepalive time.Time
}

func newSSEWriter(w http.ResponseWriter, flusher http.Flusher, ctx context.Context) *sseWriter {
	return &sseWriter{
		w:       w,
		flusher: flusher,
		ctx:     ctx,
		buf:     make([]byte, 0, 4096),
	}
}

func (s *sseWriter) SoftGone() bool { return s != nil && s.softGone }
func (s *sseWriter) LastOK() bool   { return s != nil && s.lastOK }

func (s *sseWriter) MarkSoftGone() {
	if s != nil && !s.softGone {
		s.softGone = true
		streamSoftGoneTotal.Add(1)
	}
}

// WriteBytes writes one payload with a single Flush. Soft client errors are
// swallowed (softGone=true, lastOK=false) so callers can Requeue unacked tools
// instead of aborting ReadSSE mid-envelope.
func (s *sseWriter) WriteBytes(payload []byte, force bool) error {
	if s == nil {
		return errors.New("sse writer is nil")
	}
	s.lastOK = false
	if len(payload) == 0 {
		s.lastOK = true
		return nil
	}
	if s.softGone && !force {
		return nil
	}
	if !force && s.ctx != nil {
		select {
		case <-s.ctx.Done():
			if !s.softGone {
				s.softGone = true
				streamSoftGoneTotal.Add(1)
			}
			// Keep consuming upstream so force-finish / Complete can still run.
			return nil
		default:
		}
	}
	n, err := s.w.Write(payload)
	if err == nil && n < len(payload) {
		err = errors.New("short write: connection reset by peer")
	}
	if err != nil {
		if isSoftClientWriteError(err) || (s.ctx != nil && errors.Is(err, s.ctx.Err())) {
			if !s.softGone {
				s.softGone = true
				streamSoftGoneTotal.Add(1)
			}
			// lastOK stays false → caller Requeues unacked tools/terminal.
			return nil
		}
		return err
	}
	s.flusher.Flush()
	s.lastOK = true
	streamWritesTotal.Add(1)
	streamBytesTotal.Add(uint64(len(payload)))
	return nil
}

// WriteStrings joins frames into the reusable buffer and writes once.
func (s *sseWriter) WriteStrings(frames []string, force bool) error {
	if s == nil {
		return errors.New("sse writer is nil")
	}
	if len(frames) == 0 {
		s.lastOK = true
		return nil
	}
	s.buf = s.buf[:0]
	for _, frame := range frames {
		s.buf = append(s.buf, frame...)
	}
	return s.WriteBytes(s.buf, force)
}

// WriteString is a convenience for a single SSE frame / comment.
func (s *sseWriter) WriteString(frame string, force bool) error {
	if frame == "" {
		if s != nil {
			s.lastOK = true
		}
		return nil
	}
	return s.WriteBytes([]byte(frame), force)
}

// Keepalive writes frame at most once per minInterval unless force is true and
// the interval has elapsed (or never written). Returns nil without writing when
// throttled — callers use this for pending-tool warmth, not terminal delivery.
func (s *sseWriter) Keepalive(frame string, minInterval time.Duration, force bool) error {
	if s == nil {
		return errors.New("sse writer is nil")
	}
	if frame == "" {
		return nil
	}
	if minInterval <= 0 {
		minInterval = 1500 * time.Millisecond
	}
	now := time.Now()
	if !s.lastKeepalive.IsZero() && now.Sub(s.lastKeepalive) < minInterval {
		return nil
	}
	if err := s.WriteString(frame, force); err != nil {
		return err
	}
	if s.lastOK {
		s.lastKeepalive = now
		streamKeepalivesTotal.Add(1)
	}
	return nil
}

// DefaultKeepaliveInterval is the minimum gap between forced pending-tool
// keepalives. Reverse proxies cut around 30–60s idle; 1.5s keeps the pipe warm
// without one Flush per tool-arg micro-chunk.
const DefaultKeepaliveInterval = 1500 * time.Millisecond

// Cheap SSE frame classifiers used by stream groupers (avoid re-scanning
// full JSON with ad-hoc strings.Contains at every call site).

func frameIsAnthropicToolStart(frame string) bool {
	return strings.Contains(frame, `"tool_use"`) && strings.Contains(frame, "content_block_start")
}

func frameIsResponsesToolStart(frame string) bool {
	return strings.Contains(frame, "function_call") && strings.Contains(frame, "response.output_item.added")
}

func frameIsAnthropicTerminal(frame string) bool {
	return strings.Contains(frame, "message_stop") || strings.Contains(frame, "message_delta") ||
		strings.Contains(frame, `"type":"message_stop"`) || strings.Contains(frame, `"type":"message_delta"`)
}

func frameIsResponsesTerminal(frame string) bool {
	return strings.Contains(frame, "response.completed") || strings.Contains(frame, "response.failed") ||
		strings.Contains(frame, "[DONE]")
}

func frameNeedsAnthropicImmediate(frame string) bool {
	return strings.Contains(frame, `"tool_use"`) ||
		strings.Contains(frame, "message_stop") ||
		strings.Contains(frame, "message_delta") ||
		strings.Contains(frame, "message_start") ||
		strings.Contains(frame, "event: error") ||
		strings.Contains(frame, `"type":"error"`)
}

func frameNeedsResponsesImmediate(frame string) bool {
	return strings.Contains(frame, "function_call") ||
		strings.Contains(frame, "response.completed") ||
		strings.Contains(frame, "response.failed") ||
		strings.Contains(frame, "[DONE]") ||
		strings.Contains(frame, "response.output_item.added")
}

// waitToolGap sleeps for gap or until ctx is done. Returns true if ctx cancelled.
// Uses a single Timer (not time.After) so multi-tool turns do not leak timers
// when the client soft-disconnects mid-gap.
func waitToolGap(ctx context.Context, gap time.Duration) (cancelled bool) {
	if gap <= 0 || ctx == nil {
		return false
	}
	timer := time.NewTimer(gap)
	select {
	case <-ctx.Done():
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		return true
	case <-timer.C:
		return false
	}
}

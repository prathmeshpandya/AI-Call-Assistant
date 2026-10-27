# Negotiation module: technical design (not implemented in v1)

## Goal
The goal is to let the voice assistant bargain on the user's behalf within limits the user defines. The voice pipeline doesn't change at all.

## Principles
1. **Limits are structured state, not prompt text.** The LLM never gets to "decide" a limit. Code enforces limits before anything is said.
2. **The LLM proposes, the policy disposes.** The LLM extracts offers and drafts wording. A deterministic `NegotiationPolicy` decides whether to accept, reject, counter or escalate, and at what price.
3. **No invented facts.** Prices, specs and promises may only come from `NegotiationState`, the user's configuration, or the counterparty's own words captured in history.
4. **No binding commitments without authority.** Acceptance above `requires_approval_above`, or any final agreement, needs explicit user confirmation, unless the user granted that authority in advance within defined limits.

## State (planned `negotiation/state.py`)
```python
@dataclass
class NegotiationConfig:            # set by the user before the conversation
    role: Literal["buyer", "seller"]
    item: str
    target_price: Decimal
    limit_price: Decimal            # max purchase price (buyer) / min selling price (seller)
    initial_offer: Decimal
    currency: str
    strategy: Literal["anchored", "split_difference", "fixed_steps"]
    max_concessions: int
    max_concession_step: Decimal
    walk_away: WalkAwayRule         # e.g. counterparty insists beyond limit N times
    requires_approval_above: Decimal | None
    can_accept_autonomously: bool   # default False

@dataclass
class Offer:
    party: Literal["us", "them"]
    amount: Decimal
    terms: str
    utterance_turn: int             # provenance: which turn it was heard/said in

@dataclass
class NegotiationState:
    config: NegotiationConfig
    history: list[Offer]
    concessions_made: int
    status: Literal["open", "agreed_pending_approval", "agreed", "walked_away", "escalated"]
```

## Flow per user turn
```
final transcript ─▶ OfferExtractor (LLM tool call / JSON mode) ─▶ Offer? ─▶ NegotiationState.history
                                                                     │
                                         NegotiationPolicy.decide(state) ─▶ Decision(action, amount, rationale)
                                                                     │
                       system prompt + "DECISION: counter at 420 EUR because ..." (read-only facts)
                                                                     │
                                         LLM phrases the decision ─▶ existing ResponseStreamer ─▶ TTS
                                                                     │
                                   OutputGuard: reject/regenerate if the reply states a number not in Decision
```

## Integration points with the existing code
- **`VoiceSession._start_response`:** if a `NegotiationController` is attached, call it before `build_messages`. It returns extra system context with the decision and facts, or an `escalate` signal. On escalate, the assistant tells the counterparty it needs to check with the user, and a `negotiation_approval_required` event goes to the UI.
- **`ConversationManager`:** unchanged. Offers carry `utterance_turn` provenance, so heard-only history stays the source of truth.
- **Tool calling:** `llm_base.LLMClient` gains an optional `complete_json(messages, schema)` for offer extraction. llama-server supports JSON-schema constrained output through `response_format`.
- **`ResponseStreamer`:** gains an optional `OutputGuard` hook that checks each sentence before TTS. A sentence with an unauthorized number is dropped, and generation is retried with a correction.
- **UI:** a negotiation panel to set the config, view the offer history and decision rationale, and an approve/decline control.

## Testing plan
- Unit tests for the policy: every strategy stays within limits, never concedes more than `max_concessions`, and walks away correctly.
- Property test: random offer sequences never produce a decision outside `[initial_offer, limit_price]`.
- Guard test: an LLM reply that mentions an unapproved price is blocked.

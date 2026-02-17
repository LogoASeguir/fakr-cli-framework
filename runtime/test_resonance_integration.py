"""
TEST_RESONANCE_INTEGRATION.PY

Validates PHASE 1A: Active wiring of resonance + style + embryo into model decisions.

This test suite verifies:
1. Resonance state is correctly formatted and injected
2. Style knobs are translated into actionable hints
3. Embryo weights are included in routing context
4. Full message construction includes [ROUTING_STATE] section
5. Backward compatibility (optional parameters)

Run with: python test_resonance_integration.py
"""

from typing import Dict, Optional, Any
import json


class MockAnythingLLMBackend:
    """Minimal mock for testing without network calls."""
    
    def __init__(self):
        self.last_message = None
    
    def simple_reply(self, message: str, mode: str = "chat") -> str:
        """Store message for inspection, return dummy reply."""
        self.last_message = message
        return "[Mock response from LLM]"


class TestableModelClient:
    """ModelClient with exposed _build_routing_context for testing."""
    
    def __init__(self, backend: Optional[MockAnythingLLMBackend] = None):
        self.backend = backend or MockAnythingLLMBackend()
    
    def _build_routing_context(
        self,
        resonance_state: Optional[float] = None,
        style_knobs: Optional[Dict[str, float]] = None,
        embryo_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Same implementation as ModelClient.
        Build a text description of current system state for the model.
        """
        lines = []

        if resonance_state is not None:
            # Resonance: 0=struggling, 0.5=neutral, 1.0=thriving
            if resonance_state < 0.3:
                resonance_note = (
                    "The system is STRUGGLING (recent responses were not useful). "
                    "You should provide extra clarity, ask clarifying questions, "
                    "and avoid assumptions."
                )
            elif resonance_state < 0.7:
                resonance_note = (
                    "The system is NEUTRAL. Continue standard reasoning, "
                    "but be ready to adapt if the human corrects you."
                )
            else:
                resonance_note = (
                    "The system is THRIVING (recent responses were useful). "
                    "You may be slightly more confident and offer higher-level insights."
                )
            lines.append(f"[RESONANCE] {resonance_note}")
            lines.append(f"[RESONANCE_VALUE] {resonance_state:.2f}")

        if style_knobs:
            reflection = style_knobs.get("reflection_allowed", 0.5)
            verbosity = style_knobs.get("verbosity", 0.5)
            directness = style_knobs.get("directness", 0.5)

            if reflection > 0.7:
                lines.append("[STYLE] reflection_allowed: YES—include <think> blocks and reasoning.")
            else:
                lines.append("[STYLE] reflection_allowed: NO—keep <think> blocks minimal.")

            if verbosity > 0.7:
                lines.append("[STYLE] verbosity: HIGH—provide detailed explanations.")
            elif verbosity < 0.3:
                lines.append("[STYLE] verbosity: LOW—be concise and direct.")
            else:
                lines.append("[STYLE] verbosity: MEDIUM—balanced detail and brevity.")

            if directness > 0.7:
                lines.append("[STYLE] directness: HIGH—answer directly without preamble.")
            elif directness < 0.3:
                lines.append("[STYLE] directness: LOW—be thoughtful and exploratory.")
            else:
                lines.append("[STYLE] directness: MEDIUM—direct but with context.")

        if embryo_weights:
            directness = embryo_weights.get("directness", 0.5)
            curiosity = embryo_weights.get("curiosity", 0.5)
            concreteness = embryo_weights.get("concreteness", 0.5)
            abstraction = embryo_weights.get("abstraction_tolerance", 0.5)

            lines.append(
                f"[EMBRYO] Self-tune: directness={directness:.2f}, "
                f"curiosity={curiosity:.2f}, concreteness={concreteness:.2f}, "
                f"abstraction_tolerance={abstraction:.2f}"
            )

        return "\n".join(lines) if lines else "[ROUTING_STATE] default"

    def online_solve(
        self,
        prompt: str,
        context: Optional[str] = None,
        resonance_state: Optional[float] = None,
        style_knobs: Optional[Dict[str, float]] = None,
        embryo_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """Simplified test version of online_solve."""
        # Build routing context
        routing_context = self._build_routing_context(
            resonance_state=resonance_state,
            style_knobs=style_knobs,
            embryo_weights=embryo_weights,
        )

        # Construct message the same way as real ModelClient
        if context:
            message = (
                f"[SYSTEM]\n[system prompt would go here]\n\n"
                f"[ROUTING_STATE]\n{routing_context}\n\n"
                f"[CONTEXT]\n{context}\n\n"
                f"[USER]\n{prompt}"
            )
        else:
            message = (
                f"[SYSTEM]\n[system prompt would go here]\n\n"
                f"[ROUTING_STATE]\n{routing_context}\n\n"
                f"[USER]\n{prompt}"
            )

        return self.backend.simple_reply(message, mode="chat")


# ============================================================
# TEST SUITE
# ============================================================

def test_routing_context_default():
    """Test default (no state) routing context."""
    model = TestableModelClient()
    context = model._build_routing_context()
    assert context == "[ROUTING_STATE] default"
    print("✓ test_routing_context_default passed")


def test_resonance_struggling():
    """Test STRUGGLING resonance state (< 0.3)."""
    model = TestableModelClient()
    context = model._build_routing_context(resonance_state=0.1)
    assert "[RESONANCE]" in context
    assert "STRUGGLING" in context
    assert "0.10" in context
    assert "extra clarity" in context
    assert "clarifying questions" in context
    print("✓ test_resonance_struggling passed")


def test_resonance_neutral():
    """Test NEUTRAL resonance state (0.3-0.7)."""
    model = TestableModelClient()
    context = model._build_routing_context(resonance_state=0.5)
    assert "[RESONANCE]" in context
    assert "NEUTRAL" in context
    assert "0.50" in context
    assert "standard reasoning" in context
    print("✓ test_resonance_neutral passed")


def test_resonance_thriving():
    """Test THRIVING resonance state (> 0.7)."""
    model = TestableModelClient()
    context = model._build_routing_context(resonance_state=0.9)
    assert "[RESONANCE]" in context
    assert "THRIVING" in context
    assert "0.90" in context
    assert "higher-level insights" in context
    print("✓ test_resonance_thriving passed")


def test_style_high_reflection():
    """Test HIGH reflection allowed."""
    model = TestableModelClient()
    context = model._build_routing_context(
        style_knobs={"reflection_allowed": 0.8, "verbosity": 0.5, "directness": 0.5}
    )
    assert "[STYLE]" in context
    assert "reflection_allowed: YES" in context
    assert "<think>" in context
    print("✓ test_style_high_reflection passed")


def test_style_low_reflection():
    """Test LOW reflection allowed."""
    model = TestableModelClient()
    context = model._build_routing_context(
        style_knobs={"reflection_allowed": 0.2, "verbosity": 0.5, "directness": 0.5}
    )
    assert "[STYLE]" in context
    assert "reflection_allowed: NO" in context
    assert "minimal" in context
    print("✓ test_style_low_reflection passed")


def test_style_high_verbosity():
    """Test HIGH verbosity."""
    model = TestableModelClient()
    context = model._build_routing_context(
        style_knobs={"reflection_allowed": 0.5, "verbosity": 0.8, "directness": 0.5}
    )
    assert "[STYLE]" in context
    assert "verbosity: HIGH" in context
    assert "detailed" in context
    print("✓ test_style_high_verbosity passed")


def test_style_low_verbosity():
    """Test LOW verbosity."""
    model = TestableModelClient()
    context = model._build_routing_context(
        style_knobs={"reflection_allowed": 0.5, "verbosity": 0.2, "directness": 0.5}
    )
    assert "[STYLE]" in context
    assert "verbosity: LOW" in context
    assert "concise" in context
    print("✓ test_style_low_verbosity passed")


def test_style_high_directness():
    """Test HIGH directness."""
    model = TestableModelClient()
    context = model._build_routing_context(
        style_knobs={"reflection_allowed": 0.5, "verbosity": 0.5, "directness": 0.8}
    )
    assert "[STYLE]" in context
    assert "directness: HIGH" in context
    assert "directly without preamble" in context
    print("✓ test_style_high_directness passed")


def test_style_low_directness():
    """Test LOW directness."""
    model = TestableModelClient()
    context = model._build_routing_context(
        style_knobs={"reflection_allowed": 0.5, "verbosity": 0.5, "directness": 0.2}
    )
    assert "[STYLE]" in context
    assert "directness: LOW" in context
    assert "thoughtful" in context
    print("✓ test_style_low_directness passed")


def test_embryo_weights():
    """Test embryo weights included."""
    model = TestableModelClient()
    context = model._build_routing_context(
        embryo_weights={
            "directness": 0.7,
            "curiosity": 0.4,
            "concreteness": 0.8,
            "abstraction_tolerance": 0.3,
        }
    )
    assert "[EMBRYO]" in context
    assert "directness=0.70" in context
    assert "curiosity=0.40" in context
    assert "concreteness=0.80" in context
    assert "abstraction_tolerance=0.30" in context
    print("✓ test_embryo_weights passed")


def test_combined_resonance_style_embryo():
    """Test all three systems together."""
    model = TestableModelClient()
    context = model._build_routing_context(
        resonance_state=0.8,
        style_knobs={
            "reflection_allowed": 0.6,
            "verbosity": 0.8,
            "directness": 0.5,
        },
        embryo_weights={
            "directness": 0.5,
            "curiosity": 0.6,
            "concreteness": 0.4,
            "abstraction_tolerance": 0.7,
        },
    )
    # All three sections should appear
    assert "[RESONANCE]" in context
    assert "THRIVING" in context
    assert "[STYLE]" in context
    assert "verbosity: HIGH" in context
    assert "[EMBRYO]" in context
    assert "curiosity=0.60" in context
    print("✓ test_combined_resonance_style_embryo passed")


def test_full_message_construction():
    """Test that routing context is properly injected in full message."""
    mock_backend = MockAnythingLLMBackend()
    model = TestableModelClient(backend=mock_backend)
    
    model.online_solve(
        prompt="What is 2+2?",
        context="Math education context.",
        resonance_state=0.7,
        style_knobs={"reflection_allowed": 0.5, "verbosity": 0.5, "directness": 0.5},
        embryo_weights={"directness": 0.5, "curiosity": 0.5, "concreteness": 0.5, "abstraction_tolerance": 0.5},
    )
    
    # Check the message structure
    message = mock_backend.last_message
    assert "[SYSTEM]" in message
    assert "[ROUTING_STATE]" in message
    assert "[RESONANCE]" in message
    assert "[STYLE]" in message
    assert "[EMBRYO]" in message
    assert "[CONTEXT]" in message
    assert "Math education context." in message
    assert "[USER]" in message
    assert "What is 2+2?" in message
    
    print("✓ test_full_message_construction passed")


def test_backward_compatibility():
    """Test that old calls still work (optional parameters)."""
    model = TestableModelClient()
    
    # Should not raise exception
    result = model.online_solve(prompt="Hello")
    assert result == "[Mock response from LLM]"
    
    result = model.online_solve(prompt="Hello", context="Some context")
    assert result == "[Mock response from LLM]"
    
    print("✓ test_backward_compatibility passed")


# ============================================================
# DEMO: Show different system states
# ============================================================

def demo_struggling_state():
    """Show what a struggling system's routing context looks like."""
    print("\n" + "=" * 70)
    print("DEMO: STRUGGLING STATE (resonance=0.1, user is confused)")
    print("=" * 70)
    
    model = TestableModelClient()
    context = model._build_routing_context(
        resonance_state=0.1,
        style_knobs={
            "reflection_allowed": 0.3,
            "verbosity": 0.8,
            "directness": 0.4,
        },
        embryo_weights={
            "directness": 0.3,
            "curiosity": 0.8,
            "concreteness": 0.9,
            "abstraction_tolerance": 0.2,
        },
    )
    print(context)


def demo_thriving_state():
    """Show what a thriving system's routing context looks like."""
    print("\n" + "=" * 70)
    print("DEMO: THRIVING STATE (resonance=0.9, user is happy)")
    print("=" * 70)
    
    model = TestableModelClient()
    context = model._build_routing_context(
        resonance_state=0.9,
        style_knobs={
            "reflection_allowed": 0.8,
            "verbosity": 0.6,
            "directness": 0.8,
        },
        embryo_weights={
            "directness": 0.8,
            "curiosity": 0.5,
            "concreteness": 0.4,
            "abstraction_tolerance": 0.9,
        },
    )
    print(context)


def demo_style_evolution():
    """Show style changes across interactions."""
    print("\n" + "=" * 70)
    print("DEMO: STYLE EVOLUTION")
    print("Turn 1: User prefers brief, direct answers")
    print("=" * 70)
    
    model = TestableModelClient()
    context = model._build_routing_context(
        resonance_state=0.5,
        style_knobs={
            "reflection_allowed": 0.2,
            "verbosity": 0.2,
            "directness": 0.9,
        },
    )
    print(context)
    
    print("\nTurn 5: User now asking deeper questions, wants reflection")
    print("-" * 70)
    context = model._build_routing_context(
        resonance_state=0.6,
        style_knobs={
            "reflection_allowed": 0.8,
            "verbosity": 0.7,
            "directness": 0.5,
        },
    )
    print(context)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 1A INTEGRATION TEST SUITE")
    print("Testing: Resonance + Style + Embryo Wiring")
    print("=" * 70)
    
    # Run all tests
    test_routing_context_default()
    test_resonance_struggling()
    test_resonance_neutral()
    test_resonance_thriving()
    test_style_high_reflection()
    test_style_low_reflection()
    test_style_high_verbosity()
    test_style_low_verbosity()
    test_style_high_directness()
    test_style_low_directness()
    test_embryo_weights()
    test_combined_resonance_style_embryo()
    test_full_message_construction()
    test_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    
    # Run demos
    demo_struggling_state()
    demo_thriving_state()
    demo_style_evolution()
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("- Resonance state properly translates to confidence modulation")
    print("- Style knobs convert to discrete behavioral hints")
    print("- Embryo weights included alongside state")
    print("- All three systems integrated into message construction")
    print("- Backward compatibility maintained")
    print("=" * 70 + "\n")

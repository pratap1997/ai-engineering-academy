"""
manim_perceptron.py — 3Blue1Brown-style Manim math animations for Module 001.

Scenes:
  1. PerceptronNeuronScene  — Animated biological neuron → Perceptron abstraction
  2. DecisionBoundaryScene  — 2D hyperplane rotating in feature space with axes
  3. WeightUpdateScene      — Vector arrow showing Δw = η(y-ŷ)x update rule
  4. XORImpossibleScene     — Proves geometrically why XOR can't be separated

Run:
  manim -pql manim_perceptron.py PerceptronNeuronScene
  manim -pql manim_perceptron.py DecisionBoundaryScene
  manim -pqh manim_perceptron.py AllScenes   # high quality
"""

from manim import *


class PerceptronNeuronScene(Scene):
    """Biological neuron → mathematical perceptron abstraction."""

    def construct(self):
        self.camera.background_color = "#090C10"

        # Title
        title = Text("The Perceptron", font_size=56, color=WHITE).to_edge(UP, buff=0.4)
        subtitle = Text("Frank Rosenblatt, 1958", font_size=28, color="#818CF8").next_to(title, DOWN, buff=0.1)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.5)

        # Input nodes
        inputs = VGroup(*[
            Circle(radius=0.38, color=BLUE_C, fill_opacity=0.3).move_to([-4.5, 1.5 - i * 1.5, 0])
            for i in range(3)
        ])
        input_labels = VGroup(*[
            MathTex(f"x_{i+1}", color=BLUE_C, font_size=36).next_to(inputs[i], LEFT, buff=0.15)
            for i in range(3)
        ])

        # Perceptron body
        neuron = Circle(radius=0.65, color="#6366F1", fill_opacity=0.25, stroke_width=3)
        neuron.move_to([0, 0, 0])
        sigma = MathTex(r"\Sigma", color=WHITE, font_size=48).move_to(neuron)

        # Output node
        output = Circle(radius=0.38, color=GREEN_C, fill_opacity=0.3).move_to([3.5, 0, 0])
        output_label = MathTex(r"\hat{y}", color=GREEN_C, font_size=40).next_to(output, RIGHT, buff=0.15)

        # Weights on arrows
        weights = ["w_1", "w_2", "w_3"]
        arrows = VGroup(*[
            Arrow(inputs[i].get_right(), neuron.get_left(), buff=0.1, color="#A5B4FC", stroke_width=2.5)
            for i in range(3)
        ])
        weight_labels = VGroup(*[
            MathTex(weights[i], color="#A5B4FC", font_size=30).next_to(arrows[i], UP, buff=0.1)
            for i in range(3)
        ])

        # Output arrow
        out_arrow = Arrow(neuron.get_right(), output.get_left(), buff=0.1, color=GREEN_C, stroke_width=3)

        # Activation label
        step_fn = MathTex(r"f(z) = \begin{cases} 1 & z \geq 0 \\ 0 & z < 0 \end{cases}",
                          color=YELLOW_C, font_size=32).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(inputs), Write(input_labels))
        self.play(Create(neuron), Write(sigma))
        self.play(Create(arrows), Write(weight_labels))
        self.play(Create(out_arrow), FadeIn(output), Write(output_label))
        self.wait(0.5)
        self.play(Write(step_fn))
        self.wait(2)


class DecisionBoundaryScene(Scene):
    """Animated hyperplane rotating to separate data classes."""

    def construct(self):
        self.camera.background_color = "#090C10"

        title = Text("Linear Decision Boundary", font_size=48, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Axes
        axes = Axes(
            x_range=[-0.5, 1.8, 0.5],
            y_range=[-0.5, 1.8, 0.5],
            x_length=7,
            y_length=7,
            axis_config={"color": GREY_C, "stroke_width": 2},
        ).move_to([0, -0.3, 0])

        labels = axes.get_axis_labels(
            MathTex("x_1", color=GREY_C, font_size=32),
            MathTex("x_2", color=GREY_C, font_size=32),
        )

        self.play(Create(axes), Write(labels))

        # Data points (AND gate)
        pos_points = VGroup(*[
            Dot(axes.c2p(x, y), color=GREEN_C, radius=0.18).set_stroke(GREEN_C, 3)
            for x, y in [(1, 1)]
        ])
        neg_points = VGroup(*[
            Dot(axes.c2p(x, y), color=RED_C, radius=0.18).set_stroke(RED_C, 3)
            for x, y in [(0, 0), (1, 0), (0, 1)]
        ])

        pos_label = MathTex(r"y=1", color=GREEN_C, font_size=28).next_to(pos_points[0], UR, buff=0.15)
        self.play(FadeIn(pos_points), FadeIn(neg_points), Write(pos_label))
        self.wait(0.5)

        # Decision boundary line (starts far from optimal, rotates to correct)
        boundary = axes.plot(
            lambda x: -0.5 * x + 1.8,
            x_range=[-0.3, 1.7],
            color="#38BDF8",
            stroke_width=4,
        )
        boundary_label = MathTex(r"w^Tx + b = 0", color="#38BDF8", font_size=32).to_edge(DOWN, buff=0.5)

        self.play(Create(boundary), Write(boundary_label))
        self.wait(0.5)

        # Animate boundary rotating to optimal
        optimal = axes.plot(
            lambda x: -1.0 * x + 1.4,
            x_range=[-0.3, 1.7],
            color="#10B981",
            stroke_width=4,
        )
        self.play(Transform(boundary, optimal), run_time=2)

        converged = Text("Converged!", font_size=40, color="#10B981").to_edge(DOWN, buff=1.2)
        self.play(Write(converged))
        self.wait(2)


class WeightUpdateScene(Scene):
    """Vector visualization of the Perceptron weight update rule."""

    def construct(self):
        self.camera.background_color = "#090C10"

        title = Text("Weight Update Rule", font_size=48, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # The equation
        eq = MathTex(
            r"\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}",
            color=WHITE,
            font_size=52,
        ).shift(UP * 1.5)

        self.play(Write(eq))
        self.wait(0.5)

        # Breakdown
        terms = VGroup(
            MathTex(r"\eta", r"= \text{learning rate}", color=YELLOW_C, font_size=36).shift(DOWN * 0.5 + LEFT * 2),
            MathTex(r"(y - \hat{y})", r"= \text{error signal}", color=RED_C, font_size=36).shift(DOWN * 1.5 + LEFT * 2),
            MathTex(r"\mathbf{x}", r"= \text{input vector}", color=BLUE_C, font_size=36).shift(DOWN * 2.5 + LEFT * 2),
        )

        for t in terms:
            self.play(FadeIn(t))
            self.wait(0.3)

        # Key insight box
        box = SurroundingRectangle(
            MathTex(r"\text{Only fires when } y \neq \hat{y}", font_size=36, color="#A5B4FC").shift(DOWN * 0.5 + RIGHT * 2),
            color="#6366F1",
            buff=0.3,
            corner_radius=0.2,
        )
        insight = MathTex(r"\text{Only fires when } y \neq \hat{y}", font_size=36, color="#A5B4FC").shift(DOWN * 0.5 + RIGHT * 2)
        self.play(Write(insight), Create(box))
        self.wait(2)


class XORImpossibleScene(Scene):
    """Geometric proof of why XOR cannot be linearly separated."""

    def construct(self):
        self.camera.background_color = "#090C10"

        title = Text("XOR: The Geometric Impossibility", font_size=44, color=RED_C).to_edge(UP, buff=0.4)
        self.play(Write(title))

        axes = Axes(
            x_range=[-0.5, 1.8, 0.5],
            y_range=[-0.5, 1.8, 0.5],
            x_length=6.5,
            y_length=6.5,
            axis_config={"color": GREY_C},
        ).move_to([0, -0.3, 0])

        self.play(Create(axes))

        # XOR points: (0,0)→0, (1,1)→0 RED; (0,1)→1, (1,0)→1 GREEN
        red_pts  = VGroup(*[Dot(axes.c2p(x, y), color=RED_C,   radius=0.2) for x, y in [(0, 0), (1, 1)]])
        green_pts= VGroup(*[Dot(axes.c2p(x, y), color=GREEN_C, radius=0.2) for x, y in [(0, 1), (1, 0)]])
        labels   = VGroup(
            *[MathTex(f"({x},{y})", color=RED_C,   font_size=26).next_to(axes.c2p(x, y), UR, buff=0.15) for x, y in [(0, 0), (1, 1)]],
            *[MathTex(f"({x},{y})", color=GREEN_C, font_size=26).next_to(axes.c2p(x, y), UR, buff=0.15) for x, y in [(0, 1), (1, 0)]],
        )

        self.play(FadeIn(red_pts), FadeIn(green_pts), Write(labels))
        self.wait(0.5)

        # Show a line trying and failing — oscillate between angles
        line = axes.plot(lambda x: -x + 1.2, x_range=[-0.3, 1.7], color=YELLOW_C, stroke_width=3)
        self.play(Create(line))

        fail_text = Text("Cannot separate!", font_size=36, color=RED_C).shift(DOWN * 3.2)
        self.play(Write(fail_text))

        for slope in [0.5, -2.0, 1.5, -0.8]:
            new_line = axes.plot(lambda x, s=slope: s * x + 0.4, x_range=[-0.3, 1.7], color=YELLOW_C, stroke_width=3)
            self.play(Transform(line, new_line), run_time=0.7)

        # Show MLP solution needed
        mlp_note = MathTex(
            r"\Rightarrow \text{Need Multi-Layer Perceptron (MLP)}",
            color="#A5B4FC", font_size=38
        ).shift(DOWN * 3.2)
        self.play(Transform(fail_text, mlp_note))
        self.wait(2)

# HalfWeg Algorithm

HalfWeg is an algorithm designed to train AI to solve puzzle games. The main difference from other approaches is that HalfWeg is designed for long-term planning. Instead of selecting the next immediate action, HalfWeg uses a top-down approach: it first creates high-level sub-goals, then refines these into more detailed sub-goals, and finally plans how to navigate between them.

This mirrors how humans plan. For example, if we want to fly to Tokyo, we first think about high-level goals: getting a visa, booking a hotel, and reaching the airport. Each of these is then broken down into lower-level sub-goals: to get a visa, we might need to check requirements, gather documents, and so on. At the lowest level, these sub-goals are translated into muscle movements: contracting or relaxing specific muscles. HalfWeg follows a similar hierarchical structure.

## Links

* Paper: https://arxiv.org/abs/2504.04366 - Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks

* Interactive demo: https://levinson.ai/games/sokoban

## This repository

To see train / evaluation scripts please go to [src folder](src/)

## Showcase

Let's demonstrate this using the [Sokoban](https://en.wikipedia.org/wiki/Sokoban) game:

![Sokoban example level](assets/Sokoban%20example%20level.png)

Wiki: Sokoban is a puzzle video game in which the player pushes boxes around in a warehouse, trying to get them to storage locations.

In the example above, the puzzle is represented as a 10x10 grid. Each cell can contain an empty space, the player, a box, a goal, or a wall. The player can perform four actions: move up, down, left, or right.

Top-down planning in Sokoban involves deciding where each box should go. In our minds, we might imagine an abstract high-level plan like this:

![Sokoban example level top plan](assets/Sokoban%20example%20level%20-%20plan1.png)

With such a plan in place, we can identify important landmark states: key intermediate states that help us reach the final goal:

![Sokoban planning landmarks](assets/Planning%20landmarks.png)

In this case, we have two key landmarks: moving the first box to its goal, then the second. These divide the puzzle into three segments: (1) reaching the first goal, (2) reaching the second, and (3) reaching the final goal state by placing the third box.

Top-down planning is inherently recursive. At each stage, we’re given a starting state and a goal state, and we generate a sequence of intermediate sub-goal states (landmarks). We then recursively plan how to move between each pair of these states. At the lowest level of the recursion, the output is a sequence of primitive actions (up, down, left, right).

In HalfWeg, each recursive step produces exactly one landmark state, located approximately at the midpoint between the current start and goal state. HalfWeg trains two neural network models:

* `ModelLandmark(state1, state2)` output is intermediate landmark state

* `ModelActions(state1, state2)` output is a sequence of 4 primitive actions (up, down, left, right, stop)

To solve a puzzle, HalfWeg generates a series of recursive calls. The first call finds the landmark halfway through the overall plan:

* Model call #1: `landmark_64 ← ModelLandmark(puzzle_start_state, puzzle_goal_state)`

The name `landmark_64` indicates that the model is trained to plan for a total of 128 actions. Thus, it splits the plan into two segments: one from `puzzle_start_state` to `landmark_64`, and another from `landmark_64` to `puzzle_goal_state`. Each is planned with 64 actions.

Subsequent recursive calls further split the problem:

* Model call #2: `landmark_32 ← ModelLandmark(puzzle_start_state, landmark_64)`

And so on, few more examples of the calls inside of this hierarchy of recursive calls:

* Model call #3: `landmark_16 ← ModelLandmark(puzzle_start_state, landmark_32)`

* Model call #4: `landmark_8 ← ModelLandmark(puzzle_start_state, landmark_16)`

* Model call #5: `landmark_4 ← ModelLandmark(puzzle_start_state, landmark_8)`

* Model call #6: `landmark_12 ← ModelLandmark(landmark_8, landmark_16)`

* Model call #17: `landmark_96 ← ModelLandmark(landmark_64, puzzle_goal_state)`

* Model call #19: `landmark_72 ← ModelLandmark(landmark_64, landmark_80)`

* Model call #31: `landmark_124 ← ModelLandmark(landmark_120, puzzle_goal_state)`

The `ModelLandmark` model is used to generate intermediate states. For the lowest-level planning, HalfWeg uses a different model: `ModelActions`, which directly outputs a sequence of 4 primitive actions. It is used to transition between `landmark_4*i` and `landmark_4*(i+1)` states. To generate shorter sequences, the model uses a special `stop` action.

Here is a diagram showing the full hierarchy of recursive calls used to plan 128 actions:

![HalfWeg hierarchy of calls](assets/HalfWeg%20hierarchy%20of%20calls.png)

In this image, `start` and `target` nodes (provided by the user) represent the initial and desired puzzle states (with all boxes placed on goal tiles). Circular nodes represent landmark states generated by `ModelLandmark`. Each such node has two parent nodes, corresponding to the inputs. Model call #19 is highlighted (green circles), corresponding to: `landmark_72 ← ModelLandmark(landmark_64, landmark_80)`

At the bottom of the diagram, you can see how `ModelActions` takes two states and produces a sequence of actions.

Let’s examine what happens after executing the first sequence of actions leading to `landmark_12`, which was generated by model call #6:

![Planning example 1](assets/Sokoban%20planning%20example%201.png)

At `landmark_12`, the model planned to push the first box downward in 12 steps. HalfWeg generates `landmark_12` from the following top-down call sequence:

* Model call #1: `landmark_64 ← ModelLandmark(puzzle_start_state, puzzle_goal_state)`

* Model call #2: `landmark_32 ← ModelLandmark(puzzle_start_state, landmark_64)`

* Model call #3: `landmark_16 ← ModelLandmark(puzzle_start_state, landmark_32)`

* Model call #4: `landmark_8 ← ModelLandmark(puzzle_start_state, landmark_16)`

* Model call #6: `landmark_12 ← ModelLandmark(landmark_8, landmark_16)`

You can interactively explore this puzzle here: [Interactive HalfWeg demo](https://levinson.ai/games/sokoban#ERERERHxAAAAAB8QQAAAAfEEAQARHxAIgAER8SAAAAAfEAAAAAHxAAAABB8QAAAAgfERERERHw==)

After experimenting with the demo, I observed several interesting behaviors of this six-level hierarchical planner:

* Top-level sub-goals are abstract. The model roughly indicates target box positions, but often ignores the player's position. The actual state may significantly differ from the landmark. This is similar to how a high-level human manager sets general directions without micromanaging details.

* Low-level planning is more precise regarding box movements, but still makes frequent mistakes with player positioning. This may suggest undertraining. Interestingly, this aligns with my own mental strategy: I focus on box movement and only consider the player’s position when executing the plan.

* Planning occurs in bursts. The model often produces 5–10 action bursts where `ModelActions` outputs four complete actions, followed by shorter segments of one action. This may be a robustness strategy: if the high-level assumes a sub-goal will take 7 actions, it’s safer to give the low-level model a bit more room. Thus, high-level landmarks are placed conservatively, while low-level models are encouraged to complete tasks quickly, then idle.
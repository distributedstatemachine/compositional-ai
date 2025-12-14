//! Session 17: Open Games — Rollout Demonstration
//!
//! Run with: cargo run -p compositional-games --example session17_rollout
//!
//! This example demonstrates the compositional nature of open games:
//! - Policies and environments are independent, swappable components
//! - rollout(policy, env, T) combines them to produce a trajectory
//! - Same policy, different environment = different behavior
//! - Same environment, different policy = different behavior
//!
//! Key insight: Games are boxes with typed interfaces that compose!

use compositional_games::env::{Bandit, ChainMDP, GridState, GridWorld};
use compositional_games::policy::{
    ConstantPolicy, EpsilonGreedyPolicy, FnPolicy, GreedyPolicy, RandomPolicy,
};
use compositional_games::{compare_policies, evaluate, rollout};

fn main() {
    println!("=== Session 17: Open Games — Rollout Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. Basic Rollout: Policy + Environment → Trajectory
    // -------------------------------------------------------------------------
    println!("1. Basic Rollout");
    println!("----------------\n");

    println!("The core composition: Policy × Environment → Trajectory\n");
    println!("       ┌──────────┐");
    println!("  Obs ─┤  Policy  ├─▶ Act");
    println!("       └──────────┘");
    println!("             │");
    println!("             ▼");
    println!("       ┌──────────┐");
    println!("       │   Env    │─▶ (State', Obs, Reward)");
    println!("       └──────────┘");
    println!();

    let env = GridWorld::new(5, 5);
    let policy = RandomPolicy::new(4);

    let trajectory = rollout(&policy, &env, 50);

    println!("GridWorld 5×5 with random policy:");
    println!("  Steps taken: {}", trajectory.len());
    println!("  Total reward: {:.2}", trajectory.total_reward());
    println!("  Reached goal: {}", trajectory.is_terminal());
    println!();

    // -------------------------------------------------------------------------
    // 2. Swap the Policy
    // -------------------------------------------------------------------------
    println!("2. Swap the Policy (Same Environment)");
    println!("--------------------------------------\n");

    println!("Same GridWorld, but different policies:\n");

    // Random policy
    let random = RandomPolicy::new(4);
    let stats_random = evaluate(&random, &env, 50, 100);

    // Always-right policy
    let right = ConstantPolicy::new(3usize);
    let stats_right = evaluate(&right, &env, 50, 100);

    // Custom policy that tries to reach bottom-right
    let smart = FnPolicy::new(|obs: &GridState| -> usize {
        // Go right if not at right edge, else go up
        if obs.x < 4 {
            3 // right
        } else if obs.y < 4 {
            0 // up
        } else {
            3 // stay right at goal
        }
    });
    let stats_smart = evaluate(&smart, &env, 50, 100);

    println!("  Policy       | Mean Reward | Mean Length | % Terminal");
    println!("  -------------|-------------|-------------|----------");
    println!(
        "  Random       | {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_random.mean_reward,
        stats_random.mean_length,
        100.0 * stats_random.num_terminal as f64 / 50.0
    );
    println!(
        "  Always-Right | {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_right.mean_reward,
        stats_right.mean_length,
        100.0 * stats_right.num_terminal as f64 / 50.0
    );
    println!(
        "  Smart        | {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_smart.mean_reward,
        stats_smart.mean_length,
        100.0 * stats_smart.num_terminal as f64 / 50.0
    );
    println!();

    println!("Observation: Same environment, but policy choice dramatically");
    println!("affects performance. This is the power of compositionality!");
    println!();

    // -------------------------------------------------------------------------
    // 3. Swap the Environment
    // -------------------------------------------------------------------------
    println!("3. Swap the Environment (Same Policy)");
    println!("--------------------------------------\n");

    println!("Same random policy, but different environments:\n");

    // Different grid sizes
    let env_small = GridWorld::new(3, 3);
    let env_medium = GridWorld::new(5, 5);
    let env_large = GridWorld::new(10, 10);
    let env_obstacles = GridWorld::new(5, 5).with_obstacles(vec![(2, 2), (2, 3), (3, 2)]);

    let policy = RandomPolicy::with_seed(4, 12345);

    let stats_small = evaluate(&policy, &env_small, 50, 200);
    let stats_medium = evaluate(&policy, &env_medium, 50, 200);
    let stats_large = evaluate(&policy, &env_large, 50, 200);
    let stats_obs = evaluate(&policy, &env_obstacles, 50, 200);

    println!("  Environment    | Mean Reward | Mean Length | % Terminal");
    println!("  ---------------|-------------|-------------|----------");
    println!(
        "  3×3 Grid       | {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_small.mean_reward,
        stats_small.mean_length,
        100.0 * stats_small.num_terminal as f64 / 50.0
    );
    println!(
        "  5×5 Grid       | {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_medium.mean_reward,
        stats_medium.mean_length,
        100.0 * stats_medium.num_terminal as f64 / 50.0
    );
    println!(
        "  10×10 Grid     | {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_large.mean_reward,
        stats_large.mean_length,
        100.0 * stats_large.num_terminal as f64 / 50.0
    );
    println!(
        "  5×5 w/Obstacles| {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_obs.mean_reward,
        stats_obs.mean_length,
        100.0 * stats_obs.num_terminal as f64 / 50.0
    );
    println!();

    // -------------------------------------------------------------------------
    // 4. Chain MDP Example
    // -------------------------------------------------------------------------
    println!("4. Chain MDP Example");
    println!("--------------------\n");

    let chain = ChainMDP::new(10);

    println!("Chain MDP: States 0,1,...,9. Goal is state 9.");
    println!("Actions: 0=left, 1=right\n");

    // Random policy
    let random = RandomPolicy::with_seed(2, 42);
    let stats_random = evaluate(&random, &chain, 100, 200);

    // Always-right policy (optimal)
    let right = ConstantPolicy::new(1usize);
    let stats_right = evaluate(&right, &chain, 100, 200);

    println!("  Policy      | Mean Reward | Mean Length | % Terminal");
    println!("  ------------|-------------|-------------|----------");
    println!(
        "  Random      | {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_random.mean_reward,
        stats_random.mean_length,
        100.0 * stats_random.num_terminal as f64 / 100.0
    );
    println!(
        "  Always-Right| {:>10.2} | {:>11.1} | {:>9.1}%",
        stats_right.mean_reward,
        stats_right.mean_length,
        100.0 * stats_right.num_terminal as f64 / 100.0
    );
    println!();

    // -------------------------------------------------------------------------
    // 5. Multi-Armed Bandit
    // -------------------------------------------------------------------------
    println!("5. Multi-Armed Bandit");
    println!("---------------------\n");

    let bandit = Bandit::new(vec![0.0, 0.5, 1.0, 0.3]);
    println!("4-arm bandit with means [0.0, 0.5, 1.0, 0.3]\n");

    // Random policy
    let random = RandomPolicy::with_seed(4, 123);
    let traj_random = rollout(&random, &bandit, 1000);

    // Greedy policy that knows the Q-values (oracle)
    let q_oracle = |_obs: &(), act: usize| -> f64 { [0.0, 0.5, 1.0, 0.3][act] };
    let greedy = GreedyPolicy::new(q_oracle, 4);
    let traj_greedy = rollout(&greedy, &bandit, 1000);

    // Epsilon-greedy
    let eps_greedy = EpsilonGreedyPolicy::new(q_oracle, 4, 0.1);
    let traj_eps = rollout(&eps_greedy, &bandit, 1000);

    println!("  Policy        | Total Reward (1000 pulls) | Avg/Pull");
    println!("  --------------|---------------------------|--------");
    println!(
        "  Random        | {:>25.2} | {:>7.3}",
        traj_random.total_reward(),
        traj_random.total_reward() / 1000.0
    );
    println!(
        "  Oracle Greedy | {:>25.2} | {:>7.3}",
        traj_greedy.total_reward(),
        traj_greedy.total_reward() / 1000.0
    );
    println!(
        "  ε-Greedy(0.1) | {:>25.2} | {:>7.3}",
        traj_eps.total_reward(),
        traj_eps.total_reward() / 1000.0
    );
    println!();

    println!("Optimal arm is #2 (mean=1.0)");
    println!("Oracle greedy always picks it → best performance");
    println!();

    // -------------------------------------------------------------------------
    // 6. Policy Comparison
    // -------------------------------------------------------------------------
    println!("6. Policy Comparison");
    println!("--------------------\n");

    let env = GridWorld::new(5, 5);
    let policy1 = RandomPolicy::new(4);
    let policy2 = FnPolicy::new(|obs: &GridState| -> usize {
        if obs.x < 4 {
            3
        } else if obs.y < 4 {
            0
        } else {
            3
        }
    });

    let comparison = compare_policies(&policy1, &policy2, &env, 100, 100);
    println!("{}", comparison);

    // -------------------------------------------------------------------------
    // 7. Trajectory Analysis
    // -------------------------------------------------------------------------
    println!("7. Trajectory Analysis");
    println!("----------------------\n");

    let env = GridWorld::new(5, 5);
    let policy = FnPolicy::new(|obs: &GridState| -> usize {
        // Diagonal movement: alternate right and up
        if obs.x <= obs.y {
            3 // right
        } else {
            0 // up
        }
    });

    let trajectory = rollout(&policy, &env, 20);

    println!("Diagonal policy on 5×5 grid:");
    println!("  Path: ");
    for (i, step) in trajectory.steps.iter().enumerate() {
        let action_name = match step.action {
            0 => "up",
            1 => "down",
            2 => "left",
            3 => "right",
            _ => "?",
        };
        println!(
            "    Step {}: ({},{}) --{}-> ({},{}), r={:.1}",
            i,
            step.state.x,
            step.state.y,
            action_name,
            step.next_state.x,
            step.next_state.y,
            step.reward
        );
    }
    println!();
    println!("  Total reward: {:.2}", trajectory.total_reward());
    println!(
        "  Discounted (γ=0.99): {:.2}",
        trajectory.discounted_reward(0.99)
    );
    println!();

    // -------------------------------------------------------------------------
    // 8. The Compositional Insight
    // -------------------------------------------------------------------------
    println!("8. The Compositional Insight");
    println!("----------------------------\n");

    println!("Open Games treat policies and environments as BOXES:");
    println!();
    println!("  ┌──────────────┐");
    println!("  │    Policy    │  ← Can swap without touching Env");
    println!("  └──────┬───────┘");
    println!("         │");
    println!("         ▼");
    println!("  ┌──────────────┐");
    println!("  │ Environment  │  ← Can swap without touching Policy");
    println!("  └──────────────┘");
    println!();
    println!("This is the categorical view:");
    println!("  - Policy:      Obs → Act");
    println!("  - Environment: (State, Act) → (State, Obs, Reward)");
    println!("  - Rollout:     Policy × Env → Trajectory");
    println!();
    println!("Benefits:");
    println!("  1. Test same policy on different environments");
    println!("  2. Compare different policies on same environment");
    println!("  3. Compose environments (sequential games)");
    println!("  4. Compose policies (hierarchical control)");
    println!();

    // -------------------------------------------------------------------------
    // Exercises
    // -------------------------------------------------------------------------
    println!("=== Exercises ===\n");

    println!("Exercise 1: Custom Q-function");
    println!("-----------------------------");
    // Q-function that prefers states closer to goal
    let q_custom = |obs: &GridState, act: usize| -> f64 {
        let (dx, dy): (i32, i32) = match act {
            0 => (0, 1),  // up
            1 => (0, -1), // down
            2 => (-1, 0), // left
            3 => (1, 0),  // right
            _ => (0, 0),
        };
        let new_x = (obs.x as i32 + dx).clamp(0, 4) as usize;
        let new_y = (obs.y as i32 + dy).clamp(0, 4) as usize;
        // Manhattan distance to goal (4,4), negated
        let dist = (4 - new_x) + (4 - new_y);
        -(dist as f64)
    };
    let greedy_custom = GreedyPolicy::new(q_custom, 4);
    let stats = evaluate(&greedy_custom, &GridWorld::new(5, 5), 50, 50);
    println!(
        "  Greedy with distance-based Q: mean reward = {:.2}",
        stats.mean_reward
    );
    println!();

    println!("Exercise 2: Slippery chain");
    println!("--------------------------");
    let slippery = ChainMDP::new(10).with_slip(0.3);
    let right = ConstantPolicy::new(1usize);
    let stats = evaluate(&right, &slippery, 100, 100);
    println!(
        "  Chain with 30% slip: {:.1}% reach goal (vs 100% without slip)",
        100.0 * stats.num_terminal as f64 / 100.0
    );
    println!();

    println!("Exercise 3: Reward shaping");
    println!("--------------------------");
    let env_penalty = GridWorld::new(5, 5).with_rewards(-1.0, 100.0);
    let env_no_penalty = GridWorld::new(5, 5).with_rewards(0.0, 10.0);
    let random = RandomPolicy::with_seed(4, 999);
    let stats1 = evaluate(&random, &env_penalty, 50, 100);
    let stats2 = evaluate(&random, &env_no_penalty, 50, 100);
    println!(
        "  High step penalty (-1.0): mean reward = {:.2}",
        stats1.mean_reward
    );
    println!(
        "  No step penalty (0.0):    mean reward = {:.2}",
        stats2.mean_reward
    );
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 17 Complete ===\n");

    println!("We implemented the Open Games framework:");
    println!("  • Policy trait: Obs → Act");
    println!("  • Env trait: (State, Act) → (State, Obs, Reward)");
    println!("  • rollout(policy, env, T) → Trajectory");
    println!("  • evaluate() and compare_policies() for analysis");
    println!();
    println!("Key components are SWAPPABLE:");
    println!("  • Same environment, different policies → compare strategies");
    println!("  • Same policy, different environments → test generalization");
    println!();
    println!("This is the essence of compositional game theory!");
    println!();
    println!("Next: Session 18 — Monoidal categories and string diagrams");
}

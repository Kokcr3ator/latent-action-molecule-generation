# InterDiff: Latent Action Molecule Generation - Project Documentation

## 1. The Challenge: Generating Valid & Useful Drugs

### The Context: Ligand Design
In drug discovery, the goal is often to design a small molecule (a **ligand**) that binds effectively to a specific biological target (usually a **protein**).

*   **The Target (Protein):** A large, complex biological structure (the "lock").
*   **The Ligand (Drug):** A small molecule designed to fit into the target (the "key").

### The Problem
Finding the right key is incredibly difficult because the chemical space is vast ($10^{60}$ potential molecules). We need a way to generate molecules that are not only:
1.  **Chemically Valid:** They must obey the laws of physics and chemistry (valency, ring closures, etc.).
2.  **Optimized:** They need specific properties (e.g., solubility, non-toxicity, binding affinity).

Traditional methods (like random screening) are too slow. We need **generative AI** to "imagine" new, optimized molecules.

---

## 2. The Baseline Solution: treating Chemistry as Language (Base GPT)

The first step in modern generative chemistry is to realize that molecules can be written as text strings. We use a format called **SMILES** (Simplified Molecular Input Line Entry System).

*   Ethanol: `CCO`
*   Benzene: `c1ccccc1`
*   Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`

### Enter the "Base GPT"
If chemistry is just text, we can use the same technology behind ChatGPT! We train a standard **GPT (Generative Pre-trained Transformer)** to predict the next character in a SMILES string.

*   **Input:** `C` -> `C` -> `(`
*   **Output Prediction:** `=`
*   **Result:** The model learns the "grammar" of chemistry. It can generate chemically valid molecules that look like real drugs.

### The Limitation
While a Base GPT generates *valid* molecules, it is hard to **control**.
*   If we want a molecule that is more soluble, how do we tell the GPT to do that?
*   Changing a single character (e.g., `C` to `N`) usually breaks the molecule or changes it unpredictably.
*   We lack a "steering wheel" for the generation process.

---

## 3. The Innovation: Latent Action Models (LAM)

To solve the control problem, we need to move away from generating character-by-character and start generating **concept-by-concept**.

Instead of deciding "write the letter 'C'", we want the model to decide "add a benzene ring" or "add a hydroxyl group."

### What is a Latent Action?
We don't define these actions manually. Instead, we train a neural network to **learn** them automatically. This is the **Latent Action Model (LAM)**.

*   **Concept:** The model learns a dictionary of abstract "moves" or "actions."
*   **Mechanism:**
    1.  It looks at the current state of the molecule.
    2.  It looks at the *future* state (where the molecule ends up).
    3.  It creates a discrete code (an ID number like "Action 42") that represents the transition.

Now, we have a higher-level language. Instead of spelling words letter-by-letter, we are selecting whole concepts.

---

## 4. The Full System: ControllableGPT

Now we combine everything into a powerful controllable system. The **ControllableGPT** consists of two parts working together:

1.  **The "Brain" (LAM):** Decides *what* action to take (e.g., "Action 42").
2.  **The "Hands" (Dynamics Model):** Knows *how* to write the SMILES characters for that action.

**Why is this better?**
*   **Separation of Concerns:** The "Brain" focuses on strategy (what kind of structure do we need?), while the "Hands" focus on syntax (how do I write that in SMILES?).
*   **Steerability:** It is much easier to steer the high-level actions than the low-level characters.

---

## 5. The Optimization: Reinforcement Learning (RL)

Now that we have a system that takes "actions," we can use **Reinforcement Learning (RL)** to optimize molecules for drug targets.

### The Agent (Policy)
We train an Agent (using an algorithm called **PPO**) to select actions.

### The Goal (Reward Function)
We give the Agent a "score" based on the molecule it builds.
*   **QED Score:** Is it drug-like?
*   **LogP Score:** will it dissolve in the body?
*   **Synthetic Accessibility:** Can we actually build it in a lab?

### The Loop
1.  The Agent picks a sequence of actions.
2.  The ControllableGPT builds the molecule.
3.  We check the molecule's properties.
4.  **High Score?** The Agent learns to do that again.
5.  **Low Score?** The Agent learns to avoid those actions.

Because we are optimizing *actions* (concepts) rather than *characters*, the RL training is more stable and effective than with a standard GPT.

---

## 6. Summary of Components

| Component | Role | Analogy |
|:---|:---|:---|
| **SMILES** | Text representation of molecules. | The Alphabet. |
| **Base GPT** | Generates SMILES character-by-character. | A toddler learning to spell. |
| **Latent Action Model (LAM)** | Learns discrete, high-level chemical changes. | Learning "words" instead of letters. |
| **Dynamics Model** | Executes the chemical changes in SMILES. | The hand that writes the words. |
| **ControllableGPT** | The combined system (LAM + Dynamics). | The writer. |
| **PPO (RL)** | Optimizes the choice of actions for rewards. | The editor giving feedback ("Make it simpler!"). |

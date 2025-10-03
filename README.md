# Blacksmith's Fury: An Emotionally-Driven AI NPC

Welcome to Blacksmith's Fury, an interactive web-based chat simulator featuring Borgakh, an AI blacksmith whose personality is driven by a complex, trainable emotional model. This project moves beyond traditional, scripted NPCs to create a dynamic character who remembers your actions, has a unique mood, and reacts with a full range of emotions based on established psychological principles.

This project was inspired by the concepts outlined in the academic paper, "De Ira Machinae: A Computational Architecture for an Anger-Driven Artificial Agent."

<!-- Replace with a real screenshot URL after you upload one -->
![Blacksmith's Fury Screenshot]
reenshot)
<img width="1920" height="932" alt="output" src="https://github.com/user-attachments/assets/69c09d31-66f0-4b6d-bc91-6f568c980177" />

## Core Features

### Dynamic Emotional Core
The NPC's personality is not based on simple if-then statements. It operates on a continuous, three-dimensional emotional model (Valence, Arousal, Dominance - VAD) that allows for nuanced feelings.

### Stateful Personality
Borgakh has a persistent personality with several key mechanics:

- **Memory & Grudges**: He remembers your previous provocations and will get angrier more quickly if you are persistently disrespectful.
- **Patience**: He won't get angry at the first sign of disrespect, but his patience will wear thin.
- **Mood**: Each time the server restarts, he starts with a slightly different baseline mood, making him more tolerant on some days and quicker to anger on others.

### Natural Language Understanding
The application uses a locally-run Large Language Model (Ollama with llama3.2) to interpret the player's typed sentences and classify their intent.

### Real-time Emotional Feedback
The user interface provides immediate visual feedback on the NPC's internal emotional state, with his portrait and status changing as his mood sours.

### Trainable AI Brain
The core emotional logic is powered by a Scikit-learn neural network. The AI can be retrained, modified, and improved by adjusting its training data and logic.

### Fully Local Architecture
This entire application is designed to run on your local machine without needing external, paid API keys.

## How It Works: The AI Architecture

The AI is built on a "three-brain" model where different components handle specific tasks in a pipeline.

<!-- Replace with a real flowchart image URL -->
![AI Architecture Flowchart]
<img width="1228" height="1400" alt="architectural_diagram" src="https://github.com/user-attachments/assets/78b13886-935b-4a63-b94c-5560265f69f6" />

### The Language Brain (Ollama - llama3.2)

**Role**: The NPC's "ears."

**Process**: When the player types a message, it is sent to a local Ollama server running the llama3.2 model. The model has been given a specific system prompt that instructs it to classify the user's intent into one of three structured categories: NEUTRAL, FAIRNESS_VIOLATION (insults, unfair haggling), or GOAL_OBSTRUCTION (theft, direct threats).

### The Emotional Brain (Scikit-learn Model)

**Role**: The NPC's "heart" and decision-making core.

**Process**: The structured event from the Language Brain is fed into a pre-trained neural network (`npc_model.joblib`). This model, which runs inside the Python Flask server, calculates a new emotional state (VAD values) based on the event, the NPC's current mood, and its memory of past interactions. This new emotional state determines the NPC's Finite-State Machine (FSM) state (CALM, IRRITATED, ANGRY, RAGEFUL).

### The Face & Voice (Flask & HTML/JavaScript)

**Role**: The NPC's "mouth" and "face."

**Process**: The Flask server uses the new FSM state to select an appropriate line of dialogue. It sends the dialogue, along with the raw emotional data, back to the front-end. The JavaScript on the `index.html` page then updates the chat log, the status text, and the NPC's portrait to reflect his new mood.

## Project Structure

Your project must be organized in the following way for the Flask server to work correctly:

```
/Blacksmiths_Fury/
├── app.py                  # The main Python server with all AI logic.
├── requirements.txt        # A list of all necessary Python libraries.
├── npc_model.joblib        # The trained neural network brain.*
├── npc_scaler.joblib       # The data scaler for the brain.*
├── npc_columns.joblib      # The data columns for the brain.*
├── static/
│   └── (Optional: Place local images like calm.png, angry.png here)
└── templates/
    └── index.html          # The HTML file for the user interface.
```

*Note: The .joblib files will be created automatically the first time you run `app.py`.

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- **Python**: Make sure you have Python 3.8+ installed.
- **Ollama**: You must have Ollama installed and running.
- **LLM Model**: Pull the necessary model by running this command in your terminal:

```bash
ollama pull llama3.2
```

### Installation Steps

1. **Clone the Repository**:

```bash
git clone <your-repo-url>
cd Blacksmiths_Fury
```

2. **Create a Virtual Environment**:

```bash
python -m venv game_env
source game_env/bin/activate  # On Windows, use `game_env\Scripts\activate`
```

3. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

4. **IMPORTANT - The First Run**: The pre-trained .joblib files in the repository may be incompatible with your system's library versions. Before your first run, you must delete them to allow the script to train a new, compatible brain.

```bash
rm npc_model.joblib npc_scaler.joblib npc_columns.joblib
```

## How to Run

1. **Start the AI Server**: In your terminal (with the virtual environment activated), run the main Flask application:

```bash
python app.py
```

The first time you run this after deleting the old brain, it will automatically train a new one. This may take a minute. Subsequent runs will be instantaneous.

2. **Open the Game**: Once the server is running, open your web browser and navigate to:
   
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

The chat interface should load, and you can begin your conversation with Borgakh.

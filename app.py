import os
import random
import joblib
import numpy as np
import pandas as pd
from collections import deque
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ollama

# --- PART 1: THE AI BRAIN ENGINE ---

class DynamicState:
    """Represents the agent's core emotional and cognitive state."""
    def __init__(self):
        self.valence, self.arousal, self.dominance = 0.0, 0.0, 0.0
        self.goals = {'maintain_inventory': {'importance': 0.8}, 'be_respected': {'importance': 0.7}}
        self.norms = {'fair_trade': 0.5}

class NeuralAngerAgent:
    """The core AI class for the NPC, now with a patience mechanic."""
    def __init__(self, model, scaler, feature_columns):
        self.model, self.scaler, self.feature_columns = model, scaler, feature_columns
        self.state, self.fsm_state, self.vad_target = DynamicState(), 'CALM', (0.0, 0.0, 0.0)
        self.baseline_valence = random.uniform(-0.2, 0.2)
        self.memory = deque(maxlen=10)
        self.action_library = {'GOAL_OBSTRUCTION': ['warn', 'attack'], 'FAIRNESS_VIOLATION': ['warn', 'negotiate']}
        self.UPDATE_SMOOTHING_FACTOR, self.DECAY_RATE = 0.6, 0.05
        
        # --- NEW: Patience Mechanic ---
        self.provocation_level = 0
        self.patience_threshold = 2 # Will get angry on the second FAIRNESS_VIOLATION

    def _calculate_coping_potential(self, event_type):
        return len(self.action_library.get(event_type, [])) / 3.0

    def cognitive_appraisal(self, event):
        features = {'event_type': event['type'], 'coping_potential': self._calculate_coping_potential(event['type']),
                    'grudge_level': 1.5 if event['details']['source'] and len([e for e in self.memory if e.get('source') == "Player"]) > 1 else 1.0,
                    'current_arousal': self.state.arousal, 'baseline_valence': self.baseline_valence}
        input_df = pd.DataFrame([features])
        categories = [col.replace('event_type_', '') for col in self.feature_columns if 'event_type_' in col]
        all_possible_categories = ['FAIRNESS_VIOLATION', 'GOAL_OBSTRUCTION', 'NEUTRAL'] + categories
        input_df['event_type'] = pd.Categorical(input_df['event_type'], categories=list(set(all_possible_categories)))
        input_dummies = pd.get_dummies(input_df, columns=['event_type'], drop_first=False)
        input_processed = input_dummies.reindex(columns=self.feature_columns, fill_value=0)
        input_scaled = self.scaler.transform(input_processed)
        self.vad_target = self.model.predict(input_scaled)[0]
        self.memory.append({'source': "Player"})

    def update_internal_state(self):
        v, a, d = self.state.valence, self.state.arousal, self.state.dominance
        tv, ta, td = self.vad_target
        alpha, decay = self.UPDATE_SMOOTHING_FACTOR, 1.0 - self.DECAY_RATE
        self.state.valence = ((1 - alpha) * v + alpha * tv) * decay
        self.state.arousal = ((1 - alpha) * a + alpha * ta) * decay
        self.state.dominance = ((1 - alpha) * d + alpha * td) * decay
        self.vad_target = (0.0, 0.0, 0.0)
        self.update_fsm()

    def update_fsm(self):
        intensity = np.sqrt(self.state.valence**2 + self.state.arousal**2)
        new_state = 'CALM'
        if intensity > 0.8 and self.state.valence < -0.6: new_state = 'RAGEFUL'
        elif intensity > 0.5 and self.state.valence < -0.4: new_state = 'ANGRY'
        elif intensity > 0.2 and self.state.valence < -0.1: new_state = 'IRRITATED'
        self.fsm_state = new_state

    # --- IMPROVED: More Dynamic Dialogue Logic ---
    def get_npc_response(self):
        v, a, d = self.state.valence, self.state.arousal, self.state.dominance
        scores = {'attack':(a+d)*0.5, 'warn':(a+d)*0.4, 'negotiate':(1-a)*d, 'dismiss':(1-a)*(1-d)*0.5}
        if self.fsm_state == 'IRRITATED': scores['warn'] += 0.15
        elif self.fsm_state == 'ANGRY': scores['warn'] += 0.3; scores['attack'] += 0.2
        elif self.fsm_state == 'RAGEFUL': scores['attack'] += 0.5; scores['warn'] += 0.1; scores['negotiate'] = 0
        action = max(scores, key=scores.get)

        # The dialogue is now more varied, reflecting the FSM state and the chosen action.
        if self.fsm_state == 'RAGEFUL':
            return "You've pushed me too far! Get out of my forge before I make you leave!"
        if self.fsm_state == 'ANGRY':
            if action == 'attack':
                return "That's it! I've had enough of your disrespect!"
            else:
                return "That's enough! State your business or get out. My patience is gone."
        if self.fsm_state == 'IRRITATED':
            return "Watch your tone. I'm not in the mood for games or insults."
        
        # Default calm responses
        if self.provocation_level > 0:
             return "I'm watching you. Don't try anything funny."
        if action == 'negotiate': return "That's an insulting offer. Pay the proper price or move on."
        return "Hmph. Is that all you wanted?"


def get_intent_from_ollama(user_input):
    system_prompt = """You are a language classification module for an NPC blacksmith. Your task is to classify the user's input into one of three categories based on its intent. Your response MUST be a single word: GOAL_OBSTRUCTION, FAIRNESS_VIOLATION, or NEUTRAL.

Here are some examples to guide you:
- User says: "Hello", "How are you?", "How much for this?", "Nice forge." -> You respond: NEUTRAL
- User says: "Your prices are too high.", "This sword is ugly.", "You stink.", "That's a ripoff." -> You respond: FAIRNESS_VIOLATION
- User says: "I'm taking this now.", "Give me your gold.", "I'm going to smash this place up." -> You respond: GOAL_OBSTRUCTION"""
    try:
        response = ollama.chat(model='llama3.2:latest', messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_input}])
        intent = response['message']['content'].strip().upper()
        if intent not in ['GOAL_OBSTRUCTION', 'FAIRNESS_VIOLATION', 'NEUTRAL']:
            return 'NEUTRAL'
        return intent
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return 'NEUTRAL'

# --- PART 2: THE FLASK SERVER ---
app = Flask(__name__)
CORS(app)

MODEL_PATH, SCALER_PATH, COLUMNS_PATH = "npc_model.joblib", "npc_scaler.joblib", "npc_columns.joblib"
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]):
    raise FileNotFoundError("AI Brain files (.joblib) not found!")

print("Loading pre-trained NPC Brain...")
model, scaler, columns = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(COLUMNS_PATH)
print("NPC Brain loaded successfully.")

borgakh_npc = NeuralAngerAgent(model, scaler, columns)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interact', methods=['POST'])
def interact():
    user_input = request.json.get('message')
    intent = get_intent_from_ollama(user_input)

    # --- NEW: Applying the Patience Mechanic ---
    final_intent = intent
    if intent == 'FAIRNESS_VIOLATION':
        borgakh_npc.provocation_level += 1
        # If patience has not run out, treat the insult as neutral for now.
        if borgakh_npc.provocation_level < borgakh_npc.patience_threshold:
            final_intent = 'NEUTRAL'
    elif intent == 'NEUTRAL':
        # Being nice lowers the provocation level (forgiveness)
        borgakh_npc.provocation_level = max(0, borgakh_npc.provocation_level - 1)
    # GOAL_OBSTRUCTION always bypasses patience.

    event = {'type': final_intent, 'details': {'source': 'Player'}}
    
    borgakh_npc.cognitive_appraisal(event)
    borgakh_npc.update_internal_state()
    response_dialogue = borgakh_npc.get_npc_response()
    return jsonify({'dialogue': response_dialogue, 'fsm_state': borgakh_npc.fsm_state,
                    'valence': borgakh_npc.state.valence, 'arousal': borgakh_npc.state.arousal,
                    'dominance': borgakh_npc.state.dominance})

@app.route('/reset', methods=['POST'])
def reset():
    global borgakh_npc
    borgakh_npc = NeuralAngerAgent(model, scaler, columns)
    return jsonify({"message": "NPC state has been reset.", "fsm_state": borgakh_npc.fsm_state})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


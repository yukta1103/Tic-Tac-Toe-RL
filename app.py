import streamlit as st
import numpy as np
import os
import plotly.graph_objects as go

# Import our existing modules
from src.environment import TicTacToeEnv
from src.agents import QLearningAgent, RandomAgent
from src.train import train_agent, evaluate_agent
from train_second_player import train_second_player_agent

# Page configuration
st.set_page_config(
    page_title="Tic-Tac-Toe RL",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful, modern aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Card styling with shadow */
    .game-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Cell buttons - larger and more beautiful */
    .stButton > button {
        height: 100px !important;
        font-size: 48px !important;
        font-weight: 700 !important;
        border-radius: 16px !important;
        border: 3px solid #e2e8f0 !important;
        background: #ffffff !important;
        color: #2d3748 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stButton > button:hover:not(:disabled) {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12) !important;
        border-color: #cbd5e0 !important;
        background: #f7fafc !important;
    }
    
    .stButton > button:disabled {
        background: #ffffff !important;
        border-color: #e2e8f0 !important;
        opacity: 1 !important;
    }
    
    /* Action buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        height: auto !important;
        font-size: 16px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding: 0;
        margin-bottom: 2rem;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        color: #4a5568;
        font-weight: 600;
        padding: 12px 24px;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #cbd5e0;
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border-color: transparent;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #1a1a1a;
        font-size: 32px;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #718096;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Metric container */
    [data-testid="metric-container"] {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background-color: #e2e8f0;
        border-radius: 10px;
    }
    
    /* Info/success boxes */
    .stAlert {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 10px 14px;
        font-size: 15px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Checkbox */
    .stCheckbox {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = TicTacToeEnv()
if 'agent_player1' not in st.session_state:
    model_path = 'models/agent.pkl'
    if os.path.exists(model_path):
        st.session_state.agent_player1 = QLearningAgent.load(model_path)
    else:
        st.session_state.agent_player1 = QLearningAgent()

if 'agent_player2' not in st.session_state:
    model_path = 'models/agent_second_player.pkl'
    if os.path.exists(model_path):
        st.session_state.agent_player2 = QLearningAgent.load(model_path)
    else:
        # Fallback to player 1 agent if player 2 model doesn't exist (better than random)
        if os.path.exists('models/agent.pkl'):
            st.session_state.agent_player2 = QLearningAgent.load('models/agent.pkl') 
        else:
            st.session_state.agent_player2 = QLearningAgent()

# Helper property to get current active agent
if 'agent' not in st.session_state:
    st.session_state.agent = st.session_state.agent_player1
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'playing'
if 'game_history' not in st.session_state:
    st.session_state.game_history = {'human_wins': 0, 'ai_wins': 0, 'draws': 0}
if 'board_key' not in st.session_state:
    st.session_state.board_key = 0
if 'ai_starts' not in st.session_state:
    st.session_state.ai_starts = True  # Default: AI starts

# Helper functions
def make_move(position):
    """Handle player move - Human plays as O (player -1)"""
    env = st.session_state.env
    agent = st.session_state.agent
    
    # Human move (playing as O, player -1)
    if position not in env.get_valid_actions():
        return
    
    state, reward, done, info = env.step(position)
    
    if done:
        handle_game_end(info, reward)
        return
    
    # AI move (playing as X, player 1)
    valid_actions = env.get_valid_actions()
    if len(valid_actions) > 0:
        ai_action = agent.select_action(state, valid_actions, training=False)
        state, reward, done, info = env.step(ai_action)
        
        if done:
            handle_game_end(info, reward)
    
    # Force rerun to update UI
    st.session_state.board_key += 1

def handle_game_end(info, reward):
    """Handle game end state"""
    st.session_state.game_state = 'ended'
    
    if 'winner' in info:
        if st.session_state.ai_starts:
            # AI plays as X (1), Human plays as O (-1)
            if info['winner'] == 1:
                st.session_state.game_result = "🤖 AI Wins!"
                st.session_state.game_history['ai_wins'] += 1
            else:  # winner == -1
                st.session_state.game_result = "🎉 You Win!"
                st.session_state.game_history['human_wins'] += 1
        else:
            # Human plays as X (1), AI plays as O (-1)
            if info['winner'] == 1:
                st.session_state.game_result = "🎉 You Win!"
                st.session_state.game_history['human_wins'] += 1
            else:  # winner == -1
                st.session_state.game_result = "🤖 AI Wins!"
                st.session_state.game_history['ai_wins'] += 1
    else:
        st.session_state.game_result = "🤝 Draw!"
        st.session_state.game_history['draws'] += 1

def reset_game():
    """Reset the game state"""
    st.session_state.env.reset()
    st.session_state.game_state = 'playing'
    st.session_state.game_result = None
    st.session_state.board_key += 1

# Main app
st.title("🎮 Tic-Tac-Toe RL")

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Play", "🤖 Train", "📊 Stats", "ℹ️ About"])

# PLAY TAB
with tab1:
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        # Game status
        if st.session_state.game_state == 'playing':
            if st.session_state.ai_starts:
                # AI plays first as X (1), Human plays second as O (-1)
                current_player = "AI's Turn (❌)" if st.session_state.env.current_player == 1 else "Your Turn (⭕)"
            else:
                # Human plays first as X (1), AI plays second as O (-1)
                current_player = "Your Turn (❌)" if st.session_state.env.current_player == 1 else "AI's Turn (⭕)"
            
            st.markdown(f"### {current_player}")
            
            # If AI starts and it's AI's turn, make AI move automatically
            if st.session_state.ai_starts and st.session_state.env.current_player == 1 and not st.session_state.env.is_done():
                # AI move
                valid_actions = st.session_state.env.get_valid_actions()
                if len(valid_actions) > 0:
                    ai_action = st.session_state.agent.select_action(
                        st.session_state.env.get_state(), 
                        valid_actions, 
                        training=False
                    )
                    state, reward, done, info = st.session_state.env.step(ai_action)
                    if done:
                        handle_game_end(info, reward)
                    st.session_state.board_key += 1
            
            # If Human starts and it's AI's turn (player -1), make AI move automatically
            elif not st.session_state.ai_starts and st.session_state.env.current_player == -1 and not st.session_state.env.is_done():
                # AI move
                valid_actions = st.session_state.env.get_valid_actions()
                if len(valid_actions) > 0:
                    ai_action = st.session_state.agent.select_action(
                        st.session_state.env.get_state(), 
                        valid_actions, 
                        training=False
                    )
                    state, reward, done, info = st.session_state.env.step(ai_action)
                    if done:
                        handle_game_end(info, reward)
                    st.session_state.board_key += 1
        else:
            st.markdown(f"### {st.session_state.game_result}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Render board with optimized buttons
        board = st.session_state.env.board.reshape(3, 3)
        
        for row in range(3):
            cols = st.columns(3)
            for col in range(3):
                with cols[col]:
                    cell_idx = row * 3 + col
                    cell_value = board[row, col]
                    
                    if cell_value == 1:
                        content = "❌"
                        disabled = True
                    elif cell_value == -1:
                        content = "⭕"
                        disabled = True
                    else:
                        content = "·"
                        # Disable if game ended OR if it's AI's turn
                        if st.session_state.ai_starts:
                            disabled = st.session_state.game_state != 'playing' or st.session_state.env.current_player == 1
                        else:
                            disabled = st.session_state.game_state != 'playing' or st.session_state.env.current_player == -1
                    
                    # Use unique key to prevent caching issues
                    if st.button(
                        content,
                        key=f"cell_{cell_idx}_{st.session_state.board_key}",
                        disabled=disabled,
                        use_container_width=True
                    ):
                        make_move(cell_idx)
                        st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Control button
        if st.button("🔄 New Game", use_container_width=True, type="primary"):
            reset_game()
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Game Stats")
        st.markdown("<br>", unsafe_allow_html=True)
        
        total = sum(st.session_state.game_history.values())
        if total > 0:
            human_wins = st.session_state.game_history['human_wins']
            ai_wins = st.session_state.game_history['ai_wins']
            draws = st.session_state.game_history['draws']
            
            st.metric("Human Wins", human_wins, f"{human_wins/total*100:.0f}%")
            st.metric("AI Wins", ai_wins, f"{ai_wins/total*100:.0f}%")
            st.metric("Draws", draws, f"{draws/total*100:.0f}%")
        else:
            st.info("🎮 Play some games to see your stats!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Who starts toggle
        st.markdown("### ⚙️ Settings")
        new_ai_starts = st.toggle("AI Starts First", value=st.session_state.ai_starts)
        
        if new_ai_starts != st.session_state.ai_starts:
            st.session_state.ai_starts = new_ai_starts
            # Switch active agent based on who starts
            if new_ai_starts:
                st.session_state.agent = st.session_state.agent_player1
            else:
                st.session_state.agent = st.session_state.agent_player2
            
            reset_game()
            st.rerun()

# TRAIN TAB
with tab2:
    st.markdown("### 🤖 Train New Agent")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        episodes = st.slider("Episodes", 1000, 50000, 10000, 1000)
        alpha = st.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01)
    
    with col2:
        gamma = st.slider("Discount Factor (γ)", 0.8, 0.99, 0.99, 0.01)
        epsilon = st.slider("Exploration Rate (ε)", 0.1, 0.5, 0.2, 0.05)
    
    col3, col4 = st.columns(2)
    with col3:
        epsilon_decay = st.checkbox("Enable Epsilon Decay", value=True)
    with col4:
        default_model = "agent.pkl" if st.session_state.ai_starts else "agent_second_player.pkl"
        model_name = st.text_input("Model Name", default_model)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        save_path = f"models/{model_name}"
        
        with st.spinner("Training agent... This may take a few moments."):
            if st.session_state.ai_starts:
                # Train Player 1 (X)
                agent = train_agent(
                    episodes=episodes,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    print_every=max(episodes // 10, 1),
                    save_path=save_path
                )
                st.session_state.agent_player1 = agent
            else:
                # Train Player 2 (O)
                agent = train_second_player_agent(
                    episodes=episodes,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    print_every=max(episodes // 10, 1),
                    save_path=save_path
                )
                st.session_state.agent_player2 = agent
                
            st.session_state.agent = agent
        
        progress_bar.progress(100)
        st.success("✅ Training Complete!")
        
        # Evaluate
        st.markdown("### 📊 Evaluation Results")
        player_role = 1 if st.session_state.ai_starts else -1
        stats, _ = evaluate_agent(agent, episodes=100, opponent='random', as_player=player_role)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
        with col2:
            st.metric("Loss Rate", f"{stats['loss_rate']*100:.1f}%")
        with col3:
            st.metric("Draw Rate", f"{stats['draw_rate']*100:.1f}%")
        
        st.session_state.agent = agent

# STATS TAB
with tab3:
    st.markdown("### 📊 Model Statistics")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not hasattr(st.session_state.agent, 'q_table') or len(st.session_state.agent.q_table) == 0:
        st.warning("⚠️ No trained model loaded. Train a model first!")
    else:
        stats = st.session_state.agent.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("States Learned", f"{stats['num_states']:,}")
        with col2:
            st.metric("Learning Rate", f"{stats['alpha']:.3f}")
        with col3:
            st.metric("Discount Factor", f"{stats['gamma']:.3f}")
        with col4:
            st.metric("Epsilon", f"{stats['epsilon']:.4f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎯 Evaluate Agent (100 games)", use_container_width=True, type="primary"):
            with st.spinner("Evaluating agent performance..."):
                player_role = 1 if st.session_state.ai_starts else -1
                eval_stats, _ = evaluate_agent(st.session_state.agent, episodes=100, opponent='random', as_player=player_role)
            
            st.markdown("### 🏆 Performance Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Win Rate", f"{eval_stats['win_rate']*100:.1f}%", f"{eval_stats['wins']} wins")
            with col2:
                st.metric("Loss Rate", f"{eval_stats['loss_rate']*100:.1f}%", f"{eval_stats['losses']} losses")
            with col3:
                st.metric("Draw Rate", f"{eval_stats['draw_rate']*100:.1f}%", f"{eval_stats['draws']} draws")
            
            # Chart
            fig = go.Figure(data=[
                go.Bar(
                    name='Wins', 
                    x=['Performance'], 
                    y=[eval_stats['wins']], 
                    marker_color='#667eea',
                    text=[eval_stats['wins']],
                    textposition='auto',
                ),
                go.Bar(
                    name='Losses', 
                    x=['Performance'], 
                    y=[eval_stats['losses']], 
                    marker_color='#f093fb',
                    text=[eval_stats['losses']],
                    textposition='auto',
                ),
                go.Bar(
                    name='Draws', 
                    x=['Performance'], 
                    y=[eval_stats['draws']], 
                    marker_color='#4facfe',
                    text=[eval_stats['draws']],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                barmode='group',
                height=400,
                template='plotly_white',
                showlegend=True,
                font=dict(family="Inter, sans-serif", size=14),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

# ABOUT TAB
with tab4:
    st.markdown("### ℹ️ About Tic-Tac-Toe RL")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    #### 🎮 What is this?
    An interactive web app for training and playing against a **Reinforcement Learning agent** 
    that learns Tic-Tac-Toe using **Q-Learning**.
    
    #### 🧠 How it works
    - **Q-Learning**: Agent learns optimal moves through trial and error
    - **Epsilon-Greedy**: Balances exploration vs exploitation
    - **Self-Improvement**: Gets better with more training episodes
    
    #### 🎯 Features
    - 🎮 Play against trained AI with instant response
    - 🤖 Train custom agents with adjustable parameters
    - 📊 View detailed performance statistics
    - 💾 Save and load trained models
    
    #### 📚 Q-Learning Algorithm
    The agent uses the Bellman equation:
    ```
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    ```
    
    **Parameters:**
    - **α (alpha)**: Learning rate - how much to update Q-values
    - **γ (gamma)**: Discount factor - importance of future rewards
    - **ε (epsilon)**: Exploration rate - probability of random action
    
    #### 🏆 Expected Performance
    After 10,000+ training episodes:
    - **Win Rate**: 85-95% against random opponent
    - **Loss Rate**: 0-5%
    - **Draw Rate**: 5-15%
    
    ---
    
    **Built with**: Python • Streamlit • NumPy • Plotly
    
    **Author**: yukta1103
    """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 14px;'>"
    "Made with ❤️ using Streamlit | Tic-Tac-Toe RL Project"
    "</div>",
    unsafe_allow_html=True
)

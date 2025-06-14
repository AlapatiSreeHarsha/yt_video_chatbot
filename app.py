import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer & Chatbot",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if 'video_summary' not in st.session_state:
    st.session_state.video_summary = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Hardcoded API key
API_KEY = "YOUR_API_KEY"

def extract_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    try:
        parsed_url = urlparse(youtube_url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/')[2]
        elif parsed_url.hostname in ['youtu.be']:
            return parsed_url.path[1:]
        return None
    except:
        return None

def get_video_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None

def initialize_chatbot(api_key, video_summary=None):
    """Initialize the chatbot with memory"""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Create custom prompt template
        if video_summary:
            template = f"""
            You are a helpful AI assistant. You have access to a YouTube video summary:
            
            VIDEO SUMMARY:
            {video_summary}
            
            Based on this video content and our conversation history, please provide helpful and relevant responses.
            
            Current conversation:
            {{history}}
            Human: {{input}}
            Assistant:"""
        else:
            template = """
            You are a helpful AI assistant. Please provide helpful responses based on our conversation.
            
            Current conversation:
            {history}
            Human: {input}
            Assistant:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        conversation_chain = ConversationChain(
            llm=llm,
            memory=st.session_state.memory,
            prompt=prompt,
            verbose=False
        )
        
        return conversation_chain
    
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

def summarize_video(api_key, transcript):
    """Summarize video transcript using Gemini"""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        prompt = f"""
        Please provide a comprehensive summary of this YouTube video transcript. 
        Include the main topics, key points, and important details:
        
        {transcript[:8000]}  # Limit to avoid token limits
        
        Provide a well-structured summary with:
        1. Main topic/theme
        2. Key points discussed
        3. Important conclusions or takeaways
        """
        
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        st.error(f"Error summarizing video: {str(e)}")
        return None

# Streamlit UI
st.title("🎥 YouTube Video Summarizer & Chatbot")
st.markdown("Enter a YouTube video link, get a summary, and chat about it with Gemini 2.0!")

# 1. YouTube Link Input (at the top)
youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

# Summarize Button directly after link input
summarize_clicked = st.button("Summarize Video")
if summarize_clicked and youtube_url:
    with st.spinner("Processing video..."):
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
        else:
            transcript = get_video_transcript(video_id)
            if transcript:
                summary = summarize_video(API_KEY, transcript)
                if summary:
                    st.session_state.video_summary = summary
                    st.success("Video summarized successfully!")
                    st.session_state.conversation_chain = initialize_chatbot(API_KEY, summary)

# 2. Video Summary Section (dropdown/expander)
if 'video_summary' in st.session_state and st.session_state.video_summary:
    with st.expander("📋 Show/Hide Video Summary", expanded=False):
        st.markdown(st.session_state.video_summary)

# 3. Chatbot Interface (below summary)
st.markdown("---")
st.header("💬 Chat with Gemini")

# Initialize chatbot if not already
if not st.session_state.conversation_chain and (st.session_state.video_summary or True):
    st.session_state.conversation_chain = initialize_chatbot(API_KEY, st.session_state.video_summary)

# Chat interface
if st.session_state.conversation_chain:
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                col1, col2, col3 = st.columns([2,1,7])
                with col3:
                    st.markdown(f"<div style='background-color:#A7C7E7; border-radius:10px; padding:8px 12px; margin:4px 0; text-align:right; color:#222;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
                with col1:
                    st.write("")
            else:
                col1, col2, col3 = st.columns([7,1,2])
                with col1:
                    st.markdown(f"<div style='background-color:#D1B3FF; border-radius:10px; padding:8px 12px; margin:4px 0; text-align:left; color:#222;'><b>AI:</b> {message['content']}</div>", unsafe_allow_html=True)
                with col3:
                    st.write("")
    # Input row with send button
    col_input, col_button = st.columns([8,1])
    with col_input:
        user_input = st.text_input("Type your message:", key="chat_input", placeholder="Ask about the video or anything else...", label_visibility="collapsed")
    with col_button:
        send_clicked = st.button("Send", use_container_width=True)
    # Show spinner above input if thinking
    if st.session_state.get("is_thinking", False):
        st.markdown("<div style='margin-bottom:8px;'><span>🤔 <i>Thinking...</i></span></div>", unsafe_allow_html=True)
    if send_clicked and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state["is_thinking"] = True
        st.rerun()
    # If last message is from user and is_thinking, get response
    if st.session_state.get("is_thinking", False):
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            try:
                response = st.session_state.conversation_chain.predict(input=st.session_state.chat_history[-1]["content"])
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
            st.session_state["is_thinking"] = False
            st.rerun()
else:
    st.info("Chatbot is ready! Type a message to start chatting.")

# Footer
st.markdown("---")

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import custom modules
from text_preprocessor import MultilingualPreprocessor
from topic_modeling import perform_topic_modeling
from gini_calculator import calculate_gini_per_user, calculate_gini_per_topic
from topic_evolution import analyze_general_topic_evolution
from narrative_similarity import calculate_narrative_similarity 

# --- Page Configuration ---
st.set_page_config(
    page_title="Social Media Topic Modeling System",
    page_icon="📊",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.75rem; color: #2c3e50; border-bottom: 2px solid #f0f2f6; padding-bottom: 0.3rem; margin-top: 2rem; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'custom_stopwords_text' not in st.session_state:
    st.session_state.custom_stopwords_text = ""
if "topics_info_for_sync" not in st.session_state:
    st.session_state.topics_info_for_sync = []


# --- Helper Functions ---
@st.cache_data
def create_word_cloud(_topic_model, topic_id):
    word_freq = _topic_model.get_topic(topic_id)
    if not word_freq: return None
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis", max_words=50).generate_from_frequencies(dict(word_freq))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    plt.close(fig)
    return fig



def interpret_gini(gini_score):
    # Logic is now FLIPPED for Gini Impurity
    if gini_score >= 0.6: return "🌐 Diverse Interests"
    elif gini_score >= 0.3: return "🎯 Moderately Focused"
    else: return "🔥 Highly Specialized"

# --- START OF DEFINITIVE FIX: Centralized Callback Function ---
def sync_stopwords():
    """
    This function is the single source of truth for updating stopwords.
    It's called whenever any related widget changes.
    """
    # 1. Get words from all multiselect lists
    selected_from_lists = set()
    for topic_id in st.session_state.topics_info_for_sync:
        key = f"multiselect_topic_{topic_id}"
        if key in st.session_state:
            selected_from_lists.update([s.split(' ')[0] for s in st.session_state[key]])

    # 2. Get words from the text area
    # The key for the text area is now the master state variable itself.
    typed_stopwords = set([s.strip() for s in st.session_state.custom_stopwords_text.split(',') if s])

    # 3. Combine them and update the master state variable
    combined_stopwords = typed_stopwords.union(selected_from_lists)
    st.session_state.custom_stopwords_text = ", ".join(sorted(list(combined_stopwords)))


# --- Main Page Layout ---
st.title("🌍 Multilingual Topic Modeling Dashboard")
st.markdown("Analyze textual data in multiple languages to discover topics and user trends.")

# Use a key to ensure the file uploader keeps its state, and update session_state directly
uploaded_file = st.file_uploader("Upload your CSV data", type="csv", key="csv_uploader")

# Check if a new file has been uploaded (or if it's the first time and a file exists)
if uploaded_file is not None and uploaded_file != st.session_state.get('last_uploaded_file', None):
    try:
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        st.session_state.results = None # Reset results if a new file is uploaded
        st.session_state.custom_stopwords_text = ""
        st.session_state.last_uploaded_file = uploaded_file # Store the uploaded file itself
        st.success("CSV file loaded successfully!")
    except Exception as e:
        st.error(f"Could not read CSV file. Error: {e}")
        st.session_state.df_raw = None
        st.session_state.last_uploaded_file = None

if st.session_state.df_raw is not None:
    df_raw = st.session_state.df_raw
    col1, col2, col3 = st.columns(3)

    with col1: user_id_col = st.selectbox("User ID Column", df_raw.columns, index=0, key="user_id_col")
    with col2: post_content_col = st.selectbox("Post Content Column", df_raw.columns, index=1, key="post_content_col")
    with col3: timestamp_col = st.selectbox("Timestamp Column", df_raw.columns, index=2, key="timestamp_col")
    
    st.subheader("Topic Modeling Settings")
    lang_col, topics_col = st.columns(2)
    with lang_col: language = st.selectbox("Language Model", ["english", "multilingual"], key="language_model")
    with topics_col: num_topics = st.number_input("Number of Topics", -1, help="Use -1 for automatic detection", key="num_topics")
    
    with st.expander("Advanced: Text Cleaning & Preprocessing Options", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            opts = {
                'lowercase': st.checkbox("Convert to Lowercase", True, key="opt_lowercase"),
                'lemmatize': st.checkbox("Lemmatize words", False, key="opt_lemmatize"),
                'remove_urls': st.checkbox("Remove URLs", False, key="opt_remove_urls"),
                'remove_html': st.checkbox("Remove HTML Tags", False, key="opt_remove_html")
            }
        with c2:
            opts.update({
                'remove_special_chars': st.checkbox("Remove Special Characters", False, key="opt_remove_special_chars"),
                'remove_punctuation': st.checkbox("Remove Punctuation", False, key="opt_remove_punctuation"),
                'remove_numbers': st.checkbox("Remove Numbers", False, key="opt_remove_numbers")
            })
        st.markdown("---")
        c1_emoji, c2_hashtag, c3_mention = st.columns(3)
        with c1_emoji: opts['handle_emojis'] = st.radio("Emoji Handling", ["Keep Emojis", "Remove Emojis", "Convert Emojis to Text"], index=0, key="opt_handle_emojis")
        with c2_hashtag: opts['handle_hashtags'] = st.radio("Hashtag (#) Handling", ["Keep Hashtags", "Remove Hashtags", "Extract Hashtags"], index=0, key="opt_handle_hashtags")
        with c3_mention: opts['handle_mentions'] = st.radio("Mention (@) Handling", ["Keep Mentions", "Remove Mentions", "Extract Mentions"], index=0, key="opt_handle_mentions")
        st.markdown("---")
        opts['remove_stopwords'] = st.checkbox("Remove Stopwords", True, key="opt_remove_stopwords")
        
        st.text_area(
            "Custom Stopwords (comma-separated)",
            key="custom_stopwords_text", # This one already had a key
            on_change=sync_stopwords
        )
        opts['custom_stopwords'] = [s.strip().lower() for s in st.session_state.custom_stopwords_text.split(',') if s]

    st.divider()
    process_button = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
else:
    process_button = False

st.divider()

# --- Main Processing Logic ---
if process_button:
    st.session_state.results = None
    with st.spinner("Processing your data... This may take a few minutes."):
        try:
            df = df_raw[[user_id_col, post_content_col, timestamp_col]].copy()
            df.columns = ['user_id', 'post_content', 'timestamp']
            df.dropna(subset=['user_id', 'post_content', 'timestamp'], inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if opts['handle_hashtags'] == 'Extract Hashtags': df['hashtags'] = df['post_content'].str.findall(r'#\w+')
            if opts['handle_mentions'] == 'Extract Mentions': df['mentions'] = df['post_content'].str.findall(r'@\w+')
            
            # 1. Capture the user's actual choice about stopwords
            user_wants_stopwords_removed = opts.get("remove_stopwords", False)
            custom_stopwords_list = opts.get("custom_stopwords", [])
            
            # 2. Tell the preprocessor to KEEP stopwords in the text.
            opts_for_preprocessor = opts.copy()
            opts_for_preprocessor['remove_stopwords'] = False
            
            st.info("⚙️ Initializing preprocessor and cleaning text (keeping stopwords for now)...")
            preprocessor = MultilingualPreprocessor(language=language)
            df['processed_content'] = preprocessor.preprocess_series(
                df['post_content'], 
                opts_for_preprocessor,
                n_process_spacy=1 # Keep this for stability
            )

            st.info("🔍 Performing topic modeling...")
            # FIX 3: Add the +1 logic to better target the number of topics
            if num_topics > 0:
                bertopic_nr_topics = num_topics + 1
            else:
                bertopic_nr_topics = "auto"

            docs_series = df['processed_content'].fillna('').astype(str)
            docs_to_model = docs_series[docs_series.str.len() > 0].tolist()
            df_with_content = df[docs_series.str.len() > 0].copy()
            
            if not docs_to_model:
                st.error("❌ After preprocessing, no documents were left to analyze. Please adjust your cleaning options.")
                st.stop()

            # 3. Pass the user's choice and stopwords list to BERTopic
            topic_model, topics, probs, coherence_score = perform_topic_modeling(
                docs=docs_to_model, 
                language=language, 
                nr_topics=bertopic_nr_topics,
                remove_stopwords_bertopic=user_wants_stopwords_removed,
                custom_stopwords=custom_stopwords_list
            )

            df_with_content['topic_id'] = topics
            df_with_content['probability'] = probs
            df = pd.merge(df, df_with_content[['topic_id', 'probability']], left_index=True, right_index=True, how='left')
            df['topic_id'] = df['topic_id'].fillna(-1).astype(int)
            
            st.info("📊 Calculating user engagement metrics...")
            all_unique_topics = sorted(df[df['topic_id'] != -1]['topic_id'].unique().tolist())
            all_unique_users = sorted(df['user_id'].unique().tolist())

            gini_per_user = calculate_gini_per_user(df[['user_id', 'topic_id']], all_topics=all_unique_topics)
            gini_per_topic = calculate_gini_per_topic(df[['user_id', 'topic_id']], all_users=all_unique_users)
            
            st.info("📈 Analyzing topic evolution...")
            general_evolution = analyze_general_topic_evolution(topic_model, docs_to_model, df_with_content['timestamp'].tolist())
            
            st.session_state.results = {
                'topic_model': topic_model, 
                'topic_info': topic_model.get_topic_info(),
                'df': df, 
                'gini_per_user': gini_per_user,
                'gini_per_topic': gini_per_topic, # FIX 2: Save gini_per_topic to session state
                'general_evolution': general_evolution, 
                'coherence_score': coherence_score
            }
            
            st.success("✅ Analysis complete!")
        except OSError as e:
            st.error(f"spaCy Model Error: Could not load model. Please run `python -m spacy download en_core_web_sm` and `python -m spacy download xx_ent_wiki_sm` from your terminal.")
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")
            st.exception(e)
# --- Display Results ---
if st.session_state.results:
    results = st.session_state.results
    df = results['df']
    topic_model = results['topic_model']
    topic_info = results['topic_info']
    
    st.markdown('<h2 class="sub-header">📋 Overview & Preprocessing</h2>', unsafe_allow_html=True)
    score_text = f"{results['coherence_score']:.3f}" if results['coherence_score'] is not None else "N/A"
    num_users = df['user_id'].nunique()
    avg_posts = len(df) / num_users if num_users > 0 else 0
    start_date, end_date = df['timestamp'].min(), df['timestamp'].max()
     # Option 1: More Compact Date Format
    if start_date.year == end_date.year:
        # If both dates are in the same year, only show year on the end date
        time_range_str = f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
    else:
        # If dates span multiple years, show year on both
        time_range_str = f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Posts", len(df))
    col2.metric("Unique Users", num_users)
    col3.metric("Avg Posts / User", f"{avg_posts:.1f}")
    col4.metric("Time Range", f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}")
    col5.metric("Topic Coherence", score_text)
    st.markdown("#### Preprocessing Results (Sample)")
    st.dataframe(df[['post_content', 'processed_content']].head())

    with st.expander("📊 Topic Model Evaluation Metrics"):
        st.write("""
        ### 🔹Coherence Score
        - measures how well the discovered topics make sense:
        - **> 0.6**: Excellent - Topics are very distinct and meaningful
        - **0.5 - 0.6**: Good - Topics are generally clear and interpretable  
        - **0.4 - 0.5**: Fair - Topics are somewhat meaningful but may overlap
        - **< 0.4**: Poor - Topics may be unclear or too similar
        
        💡 **Tip**: If coherence is low, try adjusting the number of topics or cleaning options.
        """)
    
    st.markdown('<h2 class="sub-header">🎯 Topic Visualization & Refinement</h2>', unsafe_allow_html=True)
    topic_options = topic_info[topic_info.Topic != -1].sort_values('Count', ascending=False)


 
    
    view1, view2 = st.tabs(["Word Clouds", "Interactive Word Lists & Refinement"])

    with view1:
        st.info("Visual representation of the most important words for each topic.")
        topics_to_show = topic_options.head(9)
        num_cols = 3
        cols = st.columns(num_cols)
        for i, row in enumerate(topics_to_show.itertuples()):
            with cols[i % num_cols]:
                st.markdown(f"##### Topic {row.Topic}: {row.Name}")
                fig = create_word_cloud(topic_model, row.Topic)
                if fig: st.pyplot(fig, use_container_width=True)
    
    with view2:
        st.info("Select or deselect words from the lists below to instantly update the custom stopwords list in the configuration section above.")
        topics_to_show = topic_options.head(9)
        # Store the topic IDs we are showing so the callback can find the right widgets
        st.session_state.topics_info_for_sync = [row.Topic for row in topics_to_show.itertuples()]

        num_cols = 3
        cols = st.columns(num_cols)
        
        # Calculate which words should be pre-selected in the multiselects
        current_stopwords_set = set([s.strip() for s in st.session_state.custom_stopwords_text.split(',') if s])

        for i, row in enumerate(topics_to_show.itertuples()):
            with cols[i % num_cols]:
                st.markdown(f"##### Topic {row.Topic}")
                topic_words = topic_model.get_topic(row.Topic)
                
                # The options for the multiselect, e.g., ["word1 (0.123)", "word2 (0.122)"]
                formatted_options = [f"{word} ({score:.3f})" for word, score in topic_words[:15]]
                
                # Determine the default selected values for this specific multiselect
                default_selection = []
                for formatted_word in formatted_options:
                    word_part = formatted_word.split(' ')[0]
                    if word_part in current_stopwords_set:
                        default_selection.append(formatted_word)

                st.multiselect(
                    f"Select words from Topic {row.Topic}",
                    options=formatted_options,
                    default=default_selection, # Pre-select words that are already in the list
                    key=f"multiselect_topic_{row.Topic}",
                    on_change=sync_stopwords, # The callback synchronizes everything
                    label_visibility="collapsed"
                )
    



    st.markdown('<h2 class="sub-header">📈 Topic Evolution</h2>', unsafe_allow_html=True)
    if not results['general_evolution'].empty:
        evo = results['general_evolution']
        
        
        # 1. Filter out the outlier topic (-1) and ensure Timestamp is a datetime object
        evo_filtered = evo[evo.Topic != -1].copy()
        evo_filtered['Timestamp'] = pd.to_datetime(evo_filtered['Timestamp'])
        
        if not evo_filtered.empty:
            # 2. Pivot the data to get topics as columns and aggregate frequencies
            evo_pivot = evo_filtered.pivot_table(
                index='Timestamp', 
                columns='Topic', 
                values='Frequency', 
                aggfunc='sum'
            ).fillna(0)
            
            # 3. Dynamically choose a good resampling frequency (Hourly, Daily, or Weekly)
            time_delta = evo_pivot.index.max() - evo_pivot.index.min()
            if time_delta.days > 60:
                resample_freq, freq_label = 'W', 'Weekly'
            elif time_delta.days > 5:
                resample_freq, freq_label = 'D', 'Daily'
            else:
                resample_freq, freq_label = 'H', 'Hourly'

            # Resample the data into the chosen time bins by summing up the frequencies
            evo_resampled = evo_pivot.resample(resample_freq).sum()

            # 4. Create the line chart using plotly.express.line
            # --- The main change is here: from px.area to px.line ---
            fig_evo = px.line(
                evo_resampled,
                x=evo_resampled.index,
                y=evo_resampled.columns,
                title=f"Topic Frequency Over Time ({freq_label} Line Chart)",
                labels={'value': 'Total Frequency', 'variable': 'Topic ID', 'index': 'Time'},
                height=500
            )
            # Make the topic IDs in the legend categorical for better color mapping
            fig_evo.for_each_trace(lambda t: t.update(name=str(t.name)))
            fig_evo.update_layout(legend_title_text='Topic')
            
            st.plotly_chart(fig_evo, use_container_width=True)
        else:
            st.info("No topic evolution data available to display (all posts may have been outliers).")
    else:
        st.warning("Could not compute topic evolution (requires more data points over time).")    


    


    st.markdown('<h2 class="sub-header">🧑‍🤝‍🧑 User Engagement Profile</h2>', unsafe_allow_html=True)

    # --- START OF THE CRITICAL FIX ---

    # 1. Create a new DataFrame containing ONLY posts from meaningful topics.
    df_meaningful = df[df['topic_id'] != -1].copy()

    # 2. Get post counts based on this meaningful data.
    meaningful_post_counts = df_meaningful.groupby('user_id').size().reset_index(name='post_count')

    # 3. Merge with the Gini results (which were already correctly calculated on meaningful topics).
    #    Using an 'inner' merge ensures we only consider users who have at least one meaningful post.
    user_metrics_df = pd.merge(
        meaningful_post_counts,
        results['gini_per_user'],
        on='user_id',
        how='inner'
    )

    # 4. Filter to include only users with more than one MEANINGFUL post.
    metrics_to_plot = user_metrics_df[user_metrics_df['post_count'] > 1].copy()

    total_meaningful_users = len(user_metrics_df)
    st.info(f"Displaying engagement profile for {len(metrics_to_plot)} users out of {total_meaningful_users} who contributed to meaningful topics.")

    # 5. Add jitter for better visualization (this part is the same as before)
    jitter_strength = 0.02
    metrics_to_plot['gini_jittered'] = metrics_to_plot['gini_coefficient'] + \
                                        np.random.uniform(-jitter_strength, jitter_strength, size=len(metrics_to_plot))

    # 6. Create the plot using the correctly filtered and prepared data.
    fig = px.scatter(
        metrics_to_plot,
        x='post_count',
        y='gini_jittered',
        title='User Engagement Profile (based on posts in meaningful topics)',
        labels={
            'post_count': 'Number of Posts in Meaningful Topics', # Updated label
            'gini_jittered': 'Gini Index (Topic Diversity)'
        },
        custom_data=['user_id', 'gini_coefficient']
    )
    fig.update_traces(
        marker=dict(opacity=0.5),
        hovertemplate="<b>User</b>: %{customdata[0]}<br><b>Meaningful Posts</b>: %{x}<br><b>Gini (Original)</b>: %{customdata[1]:.3f}<extra></extra>"
    )
    fig.update_yaxes(range=[-0.05, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # --- END OF THE CRITICAL FIX ---

    st.markdown('<h2 class="sub-header">👤 User Deep Dive</h2>', unsafe_allow_html=True)
    selected_user = st.selectbox("Select a User to Analyze", options=sorted(df['user_id'].unique()), key="selected_user_dropdown")

    if selected_user:
        user_df = df[df['user_id'] == selected_user]
        user_gini_info = user_metrics_df[user_metrics_df['user_id'] == selected_user].iloc[0]
        
        # Display the top-level metrics for the user first
        c1, c2 = st.columns(2)
        with c1: st.metric("Total Posts by User", len(user_df))
        with c2: st.metric("Topic Diversity (Gini)", f"{user_gini_info['gini_coefficient']:.3f}", help=interpret_gini(user_gini_info['gini_coefficient']))
        
        st.markdown("---") # Add a visual separator

        # --- START: New Two-Column Layout for Charts ---
        col1, col2 = st.columns(2)

        with col1:
            # --- Chart 1: Topic Distribution Pie Chart ---
            user_topic_counts = user_df['topic_id'].value_counts().reset_index()
            user_topic_counts.columns = ['topic_id', 'count']
            
            fig_pie = px.pie(
                user_topic_counts[user_topic_counts.topic_id != -1], 
                names='topic_id', 
                values='count', 
                title=f"Overall Topic Distribution for {selected_user}", 
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # --- Chart 2: Topic Evolution for User ---
            if len(user_df) > 1:
                user_evo_df = user_df[user_df['topic_id'] != -1].copy()
                user_evo_df['timestamp'] = pd.to_datetime(user_evo_df['timestamp'])

                if not user_evo_df.empty and user_evo_df['timestamp'].nunique() > 1:
                    user_pivot = user_evo_df.pivot_table(index='timestamp', columns='topic_id', aggfunc='size', fill_value=0)
                    
                    time_delta = user_pivot.index.max() - user_pivot.index.min()
                    if time_delta.days > 30: resample_freq = 'D'
                    elif time_delta.days > 2: resample_freq = 'H'
                    else: resample_freq = 'T'
                    
                    user_resampled = user_pivot.resample(resample_freq).sum()
                    row_sums = user_resampled.sum(axis=1)
                    user_proportions = user_resampled.div(row_sums, axis=0).fillna(0)

                    topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()
                    user_proportions.rename(columns=topic_name_map, inplace=True)
                    
                    fig_user_evo = px.area(
                        user_proportions,
                        x=user_proportions.index,
                        y=user_proportions.columns,
                        title=f"Topic Proportion Over Time for {selected_user}",
                        labels={'value': 'Topic Proportion', 'variable': 'Topic', 'index': 'Time'},
                    )
                    fig_user_evo.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_user_evo, use_container_width=True)
                else:
                    st.info("This user has no posts in meaningful topics or all posts occurred at the same time.")
            else:
                st.info("Topic evolution requires more than one post to display.")


        st.markdown("#### User's Most Recent Posts")
        user_posts_table = user_df[['post_content', 'timestamp', 'topic_id']] \
            .sort_values(by='timestamp', ascending=False) \
            .head(100)
        user_posts_table.columns = ['Post Content', 'Timestamp', 'Assigned Topic']
        st.dataframe(user_posts_table, use_container_width=True)

        with st.expander("Show User Distribution by Post Count"):
            # We use 'user_metrics_df' because it's based on meaningful posts
            post_distribution = user_metrics_df['post_count'].value_counts().reset_index()
            post_distribution.columns = ['Number of Posts', 'Number of Users']
            post_distribution = post_distribution.sort_values(by='Number of Posts')

            # Create a bar chart for the distribution
            fig_dist = px.bar(
                post_distribution,
                x='Number of Posts',
                y='Number of Users',
                title='User Distribution by Number of Meaningful Posts'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Display the raw data in a table
            st.write("Data Table: User Distribution")
            st.dataframe(post_distribution, use_container_width=True)

      
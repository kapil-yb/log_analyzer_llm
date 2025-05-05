# Able to load the pg_stat_statement along with DDL, and get analysis on longest running queries / most number of rows, indexes recommendation etc.
import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import ast
from typing import Dict, Any

# ==============================================
# Module 1: Data Loader
# ==============================================
class DataLoader:
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load data from CSV with YB-specific validation"""
        df = pd.read_csv(file_path)
        required_cols = {'queryid', 'query', 'calls', 'total_time', 'yb_latency_histogram'}
        assert required_cols.issubset(df.columns), "Missing YB-specific columns"
        return df

    @staticmethod
    def load_ddl(file_path: str) -> str:
        """Load schema DDL from SQL file"""
        with open(file_path, 'r') as f:
            return f.read()
    
# ==============================================
# Module 2: Pre-processor
# ==============================================
class YBPreprocessor:
    @staticmethod
    def parse_histogram(hist_str: str) -> Dict[str, float]:
        """Robust YB latency histogram parser"""
        try:
            hist = ast.literal_eval(hist_str) if isinstance(hist_str, str) else hist_str
            return {
                'latency_p50': float(hist.get('p50', 0)),
                'latency_p90': float(hist.get('p90', 0)),
                'latency_p99': float(hist.get('p99', 0)),
                'latency_max': float(hist.get('max', 0))
            }
        except:
            return {f'latency_{p}': 0 for p in ['p50', 'p90', 'p99', 'max']}

    @staticmethod
    def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate YB-specific performance metrics"""
        df['temp_usage'] = df['temp_blks_read'] + df['temp_blks_written']
        df['cache_hit_ratio'] = df['shared_blks_hit'] / (df['shared_blks_hit'] + df['shared_blks_read']).replace(0, 1)
        return df

# ==============================================
# Module 3: Analysis Engine
# ==============================================
class YBAnalyzer:
    @staticmethod
    def identify_costly_queries(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Rank queries by total execution time with latency percentiles"""
        return df.nlargest(top_n, 'total_time')[
            ['query', 'total_time', 'mean_time', 'latency_p99', 'rows', 'calls']
        ].sort_values('total_time', ascending=False)
    
    @staticmethod
    def identify_long_running_queries(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Rank queries by total execution time with latency percentiles"""
        return df.nlargest(top_n, 'total_time')[
            ['query', 'total_time', 'mean_time', 'latency_p99', 'rows', 'calls']
        ].sort_values('total_time', ascending=False)

    @staticmethod
    def identify_high_row_count_queries(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Rank queries by the total number of rows processed."""
        df_ranked = df.nlargest(top_n, 'rows').copy()
        df_ranked['avg_rows'] = df_ranked['rows'] / df_ranked['calls']
        return df_ranked[['query', 'calls', 'total_time', 'rows', 'avg_rows']].sort_values('rows', ascending=False)

    @staticmethod
    def detect_io_bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
        """Find queries with high IO wait times"""
        return df.nlargest(10, 'blk_read_time')[
            ['query', 'blk_read_time', 'shared_blks_read', 'cache_hit_ratio']
        ]

# ==============================================
# Module 4: LLM Integration
# ==============================================
class YBLLMAnalyst:
    PROMPT_TEMPLATE = """
    Analyze this YugaByteDB query performance with full schema context:

    === SCHEMA DDL ===
    {ddl}

    === QUERY METRICS ===
    {query_metrics}

    Provide output STRICTLY in this format:
    
    === QUERY ===
    {query_text}
    
    === ANALYSIS ===
    1. Performance Characteristics:
    - Execution pattern: {pattern}
    - Key metrics: {metrics}
    
    2. Schema Considerations:
    - Current indexes: {indexes}
    - Data distribution: {distribution}
    
    3. Optimization Suggestions:
    - Query rewrites: {rewrites}
    - Additional Index recommendations: {index_recs}
    === END ===
    """


    @classmethod
    def analyze(cls, df: pd.DataFrame, ddl: str = "") -> str:
        """Generate LLM insights with guaranteed output format"""
        results = []
        for _, row in df.head(3).iterrows():  # Analyze top 3 queries only
            try:
                truncated_query = (row['query'][:300] + '...') if len(row['query']) > 300 else row['query']
                
                response = ollama.chat(
                    model='llama3',
                    messages=[{
                        'role': 'user',
                        'content': cls.PROMPT_TEMPLATE.format(
                            ddl=ddl,
                            query_metrics=row.to_json(),
                            query_text=truncated_query,
                            pattern="[auto-detected]",
                            metrics="[auto-detected]",
                            suggestions="[auto-detected]",
                            indexes="[from DDL]",
                            distribution="[auto-detected]",
                            rewrites="[suggestions]",
                            index_recs="[specific indexes]"
                        )
                    }],
                    options={'temperature': 0.1}
                )
                results.append(response['message']['content'])
                print("response...............")
                print(response['message']['content'])
            except Exception as e:
                st.error(f"Error analyzing query: {str(e)}")
                continue
                
        return "\n\n".join(results)


# ==============================================
# Module 5: UI Presentation
# ==============================================
class YBVisualizer:
    @staticmethod
    def show_latency_heatmap(df: pd.DataFrame):
        """Interactive latency percentile visualization"""
        fig = px.imshow(
            df[['latency_p50', 'latency_p90', 'latency_p99']],
            labels={'value': 'Latency (ms)'},
            title="Query Latency Distribution"
        )
        st.plotly_chart(fig)

    @staticmethod
    def show_query_timing(df: pd.DataFrame):
        """Bar chart of query execution times"""
        fig = px.bar(
            df.head(20),
            x='query',
            y='total_time',
            hover_data=['latency_p99', 'rows'],
            title="Top 20 Queries by Execution Time"
        )
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig)

@staticmethod
def display_llm_insights(insights: str, title: str = "Optimization Recommendations"):
    """Robust display handler with error checking"""
    if not insights:
        st.warning("No analysis results available")
        return
    
    st.subheader(title)
    
    for block in insights.split("=== END ==="):
        try:
            if not block.strip():
                continue
                
            # Safely extract components
            query_text = block.split("=== QUERY ===")[1].split("=== ANALYSIS ===")[0].strip()
            analysis_text = block.split("=== ANALYSIS ===")[1].strip()
            
            with st.expander(f"🔍 Query Analysis", expanded=True):
                st.code(query_text, language='sql')
                st.markdown(analysis_text)
                
        except IndexError:
            st.warning("Malformed analysis block - showing raw content")
            st.text(block)

# ==============================================
# Main Application
# ==============================================
def main():
    st.title("YugaByte Query Analyzer")
    
        # Data Loading
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload pg_stat_statements CSV")
    with col2:
        ddl_file = st.file_uploader("Upload Schema DDL (optional)", type=['sql', 'txt'])
    
    if uploaded_file:
        df = DataLoader.load_csv(uploaded_file)
        
        # Load DDL if provided
    ddl = ""
    if ddl_file:
        ddl = ddl_file.read().decode("utf-8")

        # Pre-processing
        latency_data = df['yb_latency_histogram'].apply(YBPreprocessor.parse_histogram).apply(pd.Series)
        df = pd.concat([df, latency_data], axis=1)
        df = YBPreprocessor.add_derived_metrics(df)
        print(df.columns)
        # Analysis
        costly_queries = YBAnalyzer.identify_costly_queries(df)
        io_bottlenecks = YBAnalyzer.detect_io_bottlenecks(df)
        identify_high_row_count_queries = YBAnalyzer.identify_high_row_count_queries(df)
        
        # Visualization
        YBVisualizer.show_query_timing(costly_queries)
        YBVisualizer.show_latency_heatmap(df)
        print ("Costly queries")
        print(costly_queries)
        
        # LLM Insights
        if st.button("Get Optimization Recommendations"):
            insights = YBLLMAnalyst.analyze(costly_queries,ddl)
            display_llm_insights(insights, title="Insights for Costly Queries")
            insights = YBLLMAnalyst.analyze(identify_high_row_count_queries,ddl)
            display_llm_insights(insights, title="Insights for High Row Count Queries")
            print (insights)

if __name__ == "__main__":
    main()

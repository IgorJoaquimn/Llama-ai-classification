import pandas as pd
import os

def generate_html_report(individual_path, aggregate_path, video_pt_path, video_en_path, output_html):
    # Load data
    df_ind = pd.read_parquet(individual_path)
    df_agg = pd.read_parquet(aggregate_path)
    
    # Load video metadata
    df_video_pt = pd.read_csv(video_pt_path)
    df_video_en = pd.read_csv(video_en_path)
    df_video = pd.concat([df_video_pt, df_video_en]).drop_duplicates(subset=['video_id'])
    
    # Select metadata columns
    metadata = df_video[['video_id', 'title', 'channel_title']]
    
    # Merge individual results with metadata
    df_ind_merged = pd.merge(df_ind, metadata, on='video_id', how='left')
    
    # Prepare HTML
    html_content = """
    <html>
    <head>
        <title>AI Classification Report Comparison</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2 f2 f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .ai-related-true { background-color: #d4edda; }
            .ai-related-false { background-color: #f8d7da; }
            .metadata { color: #666; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>AI Classification Report Comparison</h1>
        
        <h2>1. Individual Prompt Results (First 50)</h2>
        <table>
            <tr>
                <th>Video Title / Channel</th>
                <th>Comment Text</th>
                <th>AI Related</th>
                <th>Topics</th>
                <th>Keywords</th>
            </tr>
    """
    
    # Fill individual table (limit to 50 for report size)
    for _, row in df_ind_merged.head(50).iterrows():
        ai_class = "ai-related-true" if row['is_ai_related'] else "ai-related-false"
        html_content += f"""
            <tr class="{ai_class}">
                <td>
                    <strong>{row['title']}</strong><br>
                    <span class="metadata">{row['channel_title']}</span>
                </td>
                <td>{row['text']}</td>
                <td>{row['is_ai_related']}</td>
                <td>{", ".join(row['topics']) if isinstance(row['topics'], (list, tuple)) else row['topics']}</td>
                <td>{", ".join(row['keywords']) if isinstance(row['keywords'], (list, tuple)) else row['keywords']}</td>
            </tr>
        """
        
    html_content += """
        </table>
        
        <h2>2. Aggregate Prompt Results (First 50)</h2>
        <table>
            <tr>
                <th>Video ID</th>
                <th>Keywords</th>
                <th>AI Related</th>
                <th>Topics</th>
            </tr>
    """
    
    # Aggregate results are grouped by batch in the output but main.py might have flattened them.
    # Let's check the first 50 rows of aggregate output.
    for _, row in df_agg.head(50).iterrows():
        ai_class = "ai-related-true" if row.get('is_ai_related') else "ai-related-false"
        html_content += f"""
            <tr class="{ai_class}">
                <td>{row['video_id']}</td>
                <td>{", ".join(row['keywords']) if isinstance(row.get('keywords'), (list, tuple)) else row.get('keywords')}</td>
                <td>{row.get('is_ai_related')}</td>
                <td>{", ".join(row['topics']) if isinstance(row.get('topics'), (list, tuple)) else row.get('topics')}</td>
            </tr>
        """
        
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Report generated: {output_html}")

if __name__ == "__main__":
    generate_html_report(
        'data/output/classification_individual.parquet',
        'data/output/classification_aggregate.parquet',
        'data/videos/ai_related_pt.csv',
        'data/videos/ai_related_en.csv',
        'report_comparison.html'
    )

import pandas as pd
import os

def generate_individual_report(individual_path, video_pt_path, video_en_path, output_html):
    # Load data
    if not os.path.exists(individual_path):
        print(f"Error: {individual_path} not found.")
        return
        
    df_ind = pd.read_parquet(individual_path)
    
    # Load video metadata
    df_video_pt = pd.read_csv(video_pt_path)
    df_video_en = pd.read_csv(video_en_path)
    df_video = pd.concat([df_video_pt, df_video_en]).drop_duplicates(subset=['video_id'])
    
    # Select metadata columns
    metadata = df_video[['video_id', 'title', 'channel_title']]
    
    # Merge individual results with metadata
    df_ind_merged = pd.merge(df_ind, metadata, on='video_id', how='left')
    
    # Select 10 different videos
    unique_videos = df_ind_merged['video_id'].unique()
    selected_videos = unique_videos[:10]
    df_selected = df_ind_merged[df_ind_merged['video_id'].isin(selected_videos)]
    
    # Sort for better presentation
    df_selected = df_selected.sort_values(['video_id', 'like_count'], ascending=[True, False])
    
    # Prepare HTML
    html_content = """
    <html>
    <head>
        <title>AI Individual Classification Report</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f4f4f9; }
            h1, h2 { color: #333; }
            .container { max-width: 1200px; margin: auto; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #007bff; color: white; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .ai-related-true { border-left: 5px solid #28a745; }
            .ai-related-false { border-left: 5px solid #dc3545; }
            .metadata { color: #666; font-size: 0.85em; }
            .tag { 
                display: inline-block; 
                padding: 2px 8px; 
                margin: 2px; 
                border-radius: 12px; 
                background: #e9ecef; 
                font-size: 0.8em; 
                color: #495057;
            }
            .video-header { 
                background: #343a40; 
                color: white; 
                padding: 10px 15px; 
                margin-top: 20px;
                border-radius: 5px 5px 0 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Individual Classification Report</h1>
            <p>Showing classification results for top comments from 10 different videos.</p>
    """
    
    for video_id in selected_videos:
        video_data = df_selected[df_selected['video_id'] == video_id]
        if video_data.empty:
            continue
            
        v_title = video_data.iloc[0]['title']
        v_channel = video_data.iloc[0]['channel_title']
        
        html_content += f"""
            <div class="video-header">
                <strong>{v_title}</strong> | <span style="color: #ccc;">{v_channel}</span>
            </div>
            <table>
                <tr>
                    <th style="width: 50%;">Comment Text</th>
                    <th>AI Related</th>
                    <th>Topics</th>
                    <th>Keywords</th>
                </tr>
        """
        
        for _, row in video_data.iterrows():
            ai_class = "ai-related-true" if row['is_ai_related'] else "ai-related-false"
            
            topics_html = "".join([f'<span class="tag">{t}</span>' for t in row['topics']]) if isinstance(row['topics'], (list, tuple)) else row['topics']
            keywords_html = "".join([f'<span class="tag">{k}</span>' for k in row['keywords']]) if isinstance(row['keywords'], (list, tuple)) else row['keywords']
            
            html_content += f"""
                <tr class="{ai_class}">
                    <td>{row['text']}</td>
                    <td>{"✅ Yes" if row['is_ai_related'] else "❌ No"}</td>
                    <td>{topics_html}</td>
                    <td>{keywords_html}</td>
                </tr>
            """
        html_content += "</table>"
        
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Report generated: {output_html}")

if __name__ == "__main__":
    generate_individual_report(
        'data/output/classification_individual.parquet',
        'data/videos/ai_related_pt.csv',
        'data/videos/ai_related_en.csv',
        'reports/report_individual.html'
    )

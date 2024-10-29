import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime
import requests
from io import StringIO

# MongoDB connection
@st.cache_resource
def init_mongo_client():
    return MongoClient(st.secrets["mongo"]["connection_string"])

def get_tier(essence):
    essence = float(essence) / 1000000  # Convert to correct scale
    if essence >= 80000:
        return "Multiverse"
    elif essence >= 40000:
        return "Galaxy"
    elif essence >= 20000:
        return "Constellation"
    elif essence >= 5000:
        return "Solar System"
    elif essence >= 4000:
        return "Comet"
    elif essence >= 1500:
        return "Star Dust"
    else:
        return "No Tier"

def load_project_essence(project_id):
    url = f"https://raw.githubusercontent.com/cryptomius/eclipsefi-utils/main/data/cosmic-essence-snapshots/{project_id}.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df['tier'] = df['essence'].apply(get_tier)
        return df
    except:
        return None

def hex_to_rgba(hex_color, opacity=0.2):
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Return RGBA string
    return f'rgba({r},{g},{b},{opacity})'

def create_sankey_data(project_dfs):
    # Create nodes (unique tiers across all projects)
    # Order tiers from highest to lowest, with "No Tier" at the bottom
    all_tiers = ["Multiverse", "Galaxy", "Constellation", "Solar System", "Comet", "Star Dust", "No Tier"]
    
    # Define colors for each tier (pastel palette)
    tier_colors = {
        "Multiverse": "#E6B3E6",      # Pastel Purple
        "Galaxy": "#B3B3E6",          # Pastel Blue-Purple
        "Constellation": "#B3D9FF",    # Pastel Blue
        "Solar System": "#B3E6B3",     # Pastel Green
        "Comet": "#FFE6B3",           # Pastel Gold
        "Star Dust": "#FFD1B3",       # Pastel Orange
        "No Tier": "#D9D9D9"          # Light Gray
    }
    
    # Create a second color dictionary with opacity for links
    tier_colors_with_opacity = {
        tier: hex_to_rgba(color, 0.2)
        for tier, color in tier_colors.items()
    }
    
    nodes = []
    node_indices = {}
    node_colors = []
    node_hovers = []  # New list for hover text
    
    # Calculate wallet counts and deltas for each project-tier combination
    tier_counts = {}
    for project, df in project_dfs:
        tier_counts[project] = df['tier'].value_counts()
    
    # Add project-specific tier nodes with hover information
    for i, (project, _) in enumerate(project_dfs):
        for tier in all_tiers:
            node_name = f"{project}-{tier}"
            nodes.append(node_name)
            node_indices[node_name] = len(nodes) - 1
            node_colors.append(tier_colors[tier])
            
            # Calculate wallet count and delta
            current_count = tier_counts[project].get(tier, 0)
            if i > 0:
                prev_project = project_dfs[i-1][0]
                prev_count = tier_counts[prev_project].get(tier, 0)
                delta = current_count - prev_count
                delta_text = f"(Δ {delta:+})" if delta != 0 else "(no change)"
            else:
                delta_text = ""
            
            hover_text = f"{project}<br>{tier}: {current_count} wallets {delta_text}"
            node_hovers.append(hover_text)
    
    # Calculate flows between projects
    source = []
    target = []
    value = []
    link_colors = []
    
    for i in range(len(project_dfs) - 1):
        curr_project, curr_df = project_dfs[i]
        next_project, next_df = project_dfs[i + 1]
        
        # Merge dataframes on wallet address
        merged = curr_df.merge(next_df, on='wallets', suffixes=('_curr', '_next'))
        
        # Calculate flows between tiers
        flows = merged.groupby(['tier_curr', 'tier_next']).size().reset_index()
        
        for _, flow in flows.iterrows():
            source_node = f"{curr_project}-{flow['tier_curr']}"
            target_node = f"{next_project}-{flow['tier_next']}"
            source.append(node_indices[source_node])
            target.append(node_indices[target_node])
            value.append(flow[0])
            # Use the color of the source tier for the link (with opacity)
            link_colors.append(tier_colors_with_opacity[flow['tier_curr']])
    
    return nodes, source, target, value, node_colors, link_colors, node_hovers

def create_tier_legend(tier_colors):
    st.markdown("""
        <style>
        .legend-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid rgba(0,0,0,0.2);
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            white-space: nowrap;
        }
        .legend-container {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            line-height: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    legend_items = []
    for tier, color in tier_colors.items():
        legend_items.append(
            f'<span class="legend-item">'
            f'<span class="legend-box" style="background-color: {color};"></span>'
            f'<span>{tier}</span>'
            f'</span>'
        )
    
    legend_html = f'<div class="legend-container">{"".join(legend_items)}</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

def create_tier_stats_table(project_dfs):
    # Calculate wallet counts and deltas for each project-tier combination
    stats_data = []
    all_tiers = ["No Tier", "Star Dust", "Comet", "Solar System", "Constellation", "Galaxy", "Multiverse"]
    tier_columns = all_tiers[1:]  # All tiers except "No Tier" for calculating totals
    
    for i, (project, df) in enumerate(project_dfs):
        row_data = {'Project': project}
        tier_counts = df['tier'].value_counts()
        
        # Calculate totals first
        total_excl_no_tier = sum(tier_counts.get(tier, 0) for tier in tier_columns)
        total_incl_no_tier = total_excl_no_tier + tier_counts.get("No Tier", 0)
        
        # Add regular tier columns
        for tier in all_tiers:
            current_count = tier_counts.get(tier, 0)
            if i > 0 and tier != "No Tier":
                prev_project_df = project_dfs[i-1][1]
                prev_count = prev_project_df['tier'].value_counts().get(tier, 0)
                delta = current_count - prev_count
                arrow = "🔴 ↓" if delta < 0 else "🟢 ↑" if delta > 0 else "⚪️ ="
                row_data[tier] = f"{current_count} ({arrow} {delta:+})"
            else:
                row_data[tier] = f"{current_count}"
        
        # Add total columns with deltas
        if i > 0:
            prev_project_df = project_dfs[i-1][1]
            prev_total_excl = sum(prev_project_df['tier'].value_counts().get(tier, 0) for tier in tier_columns)
            prev_total_incl = prev_total_excl + prev_project_df['tier'].value_counts().get("No Tier", 0)
            
            delta_excl = total_excl_no_tier - prev_total_excl
            delta_incl = total_incl_no_tier - prev_total_incl
            
            arrow_excl = "🔴 ↓" if delta_excl < 0 else "🟢 ↑" if delta_excl > 0 else "⚪️ ="
            arrow_incl = "🔴 ↓" if delta_incl < 0 else "🟢 ↑" if delta_incl > 0 else "⚪️ ="
            
            row_data["Total (excl No Tier)"] = f"<span class='total-count'>{total_excl_no_tier}</span> ({arrow_excl} {delta_excl:+})"
            row_data["Total (incl No Tier)"] = f"{total_incl_no_tier} ({arrow_incl} {delta_incl:+})"
        else:
            row_data["Total (excl No Tier)"] = f"<span class='total-count'>{total_excl_no_tier}</span>"
            row_data["Total (incl No Tier)"] = f"{total_incl_no_tier}"
        
        stats_data.append(row_data)
    
    # Create DataFrame and style it
    df_stats = pd.DataFrame(stats_data)
    
    # Add CSS for table styling
    st.markdown("""
        <style>
        .stats-table {
            font-size: 14px;
            margin-top: 20px;
        }
        .stats-table th {
            background-color: #f0f2f6;
            font-weight: bold;
            text-align: left;
            padding: 8px;
        }
        .stats-table td {
            padding: 8px;
            border-bottom: 1px solid #e6e6e6;
        }
        /* Style for Total (excl No Tier) column */
        .stats-table td:nth-last-child(2) {
            font-size: 16px;  /* Base size 14px + 2px */
            font-weight: normal;  /* Reset font weight */
        }
        .stats-table th:nth-last-child(2) {
            font-weight: bold;
            font-size: 16px;  /* Base size 14px + 2px */
        }
        /* Style for just the count number in Total (excl No Tier) */
        .stats-table .total-count {
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display the table
    st.markdown("### Project Tier Statistics", unsafe_allow_html=True)
    st.markdown(df_stats.to_html(classes='stats-table', escape=False, index=False), unsafe_allow_html=True)

def create_tier_column_chart(project_dfs):
    # Prepare data for plotting - reversed order of tiers
    all_tiers = ["Star Dust", "Comet", "Solar System", "Constellation", "Galaxy", "Multiverse"]
    
    # Define colors for each tier (using the same colors as Sankey)
    tier_colors = {
        "Star Dust": "#FFD1B3",       # Pastel Orange
        "Comet": "#FFE6B3",           # Pastel Gold
        "Solar System": "#B3E6B3",     # Pastel Green
        "Constellation": "#B3D9FF",    # Pastel Blue
        "Galaxy": "#B3B3E6",          # Pastel Blue-Purple
        "Multiverse": "#E6B3E6",      # Pastel Purple
    }
    
    fig = go.Figure()
    
    # For each project, create a trace showing all its tier counts
    for i, (project, df) in enumerate(project_dfs):
        y_values = []
        for tier in all_tiers:
            count = df[df['tier'] == tier]['tier'].count()
            y_values.append(count)
        
        fig.add_trace(go.Bar(
            name=project,
            x=all_tiers,
            y=y_values,
            marker_color=[tier_colors[tier] for tier in all_tiers],  # Color each bar by its tier
            showlegend=False,
            hovertemplate=f"{project}<br>Wallets: %{{y}}<extra></extra>"  # Custom hover label
        ))

    # Update layout
    fig.update_layout(
        title="Tier Distribution by Project",
        barmode='group',
        height=400,
        xaxis_title="Tiers",
        yaxis_title="Number of Wallets",
        showlegend=False,
        bargap=0.15,       # Gap between bars within a group
        bargroupgap=0.1    # Gap between bar groups
    )

    return fig

def main():
    st.set_page_config(layout="wide")
    
    # Add CSS to remove text-shadow from node labels
    st.markdown("""
        <style>
        .js-plotly-plot .node-label {
            text-shadow: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title first
    st.title("User Tier Flow Between Projects")
    
    # Define colors for each tier (pastel palette) - reversed order
    tier_colors = {
        "No Tier": "#D9D9D9",          # Light Gray
        "Star Dust": "#FFD1B3",       # Pastel Orange
        "Comet": "#FFE6B3",           # Pastel Gold
        "Solar System": "#B3E6B3",     # Pastel Green
        "Constellation": "#B3D9FF",    # Pastel Blue
        "Galaxy": "#B3B3E6",          # Pastel Blue-Purple
        "Multiverse": "#E6B3E6",      # Pastel Purple
    }
    
    # Create the legend
    create_tier_legend(tier_colors)
    
    # Initialize MongoDB client
    client = init_mongo_client()
    db = client.IDO
    
    # Get projects from MongoDB - add $ne: null to ensure whitelist_ido_start exists and isn't null
    projects = list(db.production.find(
        {
            "info.name": {"$not": {"$regex": "Eclipse Fi"}},
            "token.whitelist_ido_start": {"$exists": True, "$ne": None}
        },
        {"id": 1, "info.name": 1, "token.whitelist_ido_start": 1}
    ))
    
    # Sort projects by whitelist_ido_start with error handling
    try:
        projects.sort(key=lambda x: datetime.fromisoformat(x["token"]["whitelist_ido_start"].replace("Z", "+00:00")))
    except Exception as e:
        st.error(f"Error sorting projects: {str(e)}")
        return

    # Load essence data for each project
    project_data = []
    for project in projects:
        df = load_project_essence(project["id"])
        if df is not None:
            project_data.append((project["info"]["name"], df))
    
    if len(project_data) >= 2:  # Need at least 2 projects for comparison
        # Create Sankey diagram
        nodes, source, target, value, node_colors, link_colors, node_hovers = create_sankey_data(project_data)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors,
                hovertemplate="%{customdata}<extra></extra>",  # Custom hover template
                customdata=node_hovers  # Add hover data
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
                line=dict(color="black", width=0),
                line_color="rgba(255,255,255,0)"
            )
        )])
        
        fig.update_layout(
            height=550,
            margin=dict(t=0, b=0),
            showlegend=False,
        )
        
        # Update config to allow hovering
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': False,  # Hide the mode bar
                'staticPlot': False,  # Enable interactivity for hovering
            }
        )
        
        # Add the stats table below the Sankey diagram
        create_tier_stats_table(project_data)
        
        # Add spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add the column chart
        column_chart = create_tier_column_chart(project_data)
        st.plotly_chart(
            column_chart, 
            use_container_width=True,
            config={
                'displayModeBar': False
            }
        )
    else:
        st.error("Not enough projects with data to create flow analysis")

if __name__ == "__main__":
    main()
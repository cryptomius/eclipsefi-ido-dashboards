import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import defaultdict

# Page config
st.set_page_config(page_title="IDO Analytics Dashboard", layout="wide")
st.title("IDO Analytics Dashboard")

# Add cache clear button in the top right
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()

def process_participation_data(raw_df):
    idos = raw_df.drop_duplicates('IDO Name', keep='first').sort_values(['Date', 'IDO_Order'])
    idos = idos.reset_index(drop=True)
    
    participation_data = []
    
    for i in range(len(idos)):
        current_ido = idos.loc[i, 'IDO Name']
        current_date = idos.loc[i, 'Date']
        current_participants = set(raw_df[raw_df['IDO Name'] == current_ido]['Wallet Address'])
        
        previous_ido = None if i == 0 else idos.loc[i-1, 'IDO Name']
        previous_participants = set()
        if previous_ido:
            previous_participants = set(raw_df[raw_df['IDO Name'] == previous_ido]['Wallet Address'])
        
        all_historical_idos = set(idos.loc[:i-1, 'IDO Name']) if i > 0 else set()
        all_historical_participants = set()
        if all_historical_idos:
            all_historical_participants = set(raw_df[raw_df['IDO Name'].isin(all_historical_idos)]['Wallet Address'])
        
        new_users = len(current_participants - all_historical_participants)
        repeat = len(current_participants & previous_participants) if previous_ido else 0
        reactivated = len(current_participants & (all_historical_participants - previous_participants)) if previous_ido else 0
        churned = len(previous_participants - current_participants) if previous_ido else 0
        
        participation_data.append({
            "idoName": current_ido,
            "date": current_date,
            "new": new_users,
            "repeat": repeat,
            "reactivated": reactivated,
            "churned": churned,
            "total": len(current_participants)
        })
    
    return pd.DataFrame(participation_data)

def process_cohort_data(raw_df):
    idos = raw_df.drop_duplicates('IDO Name', keep='first').sort_values(['Date', 'IDO_Order'])
    idos = idos.reset_index(drop=True)
    
    cohort_data = []
    
    for i in range(len(idos)):
        cohort_ido = idos.loc[i, 'IDO Name']
        cohort_date = idos.loc[i, 'Date']
        cohort_wallets = set(raw_df[raw_df['IDO Name'] == cohort_ido]['Wallet Address'])
        initial_size = len(cohort_wallets)
        
        if initial_size == 0:
            continue
        
        for n, subsequent_idx in enumerate(range(i, len(idos)), 1):
            subsequent_ido = idos.loc[subsequent_idx, 'IDO Name']
            subsequent_participants = set(raw_df[raw_df['IDO Name'] == subsequent_ido]['Wallet Address'])
            retention = len(cohort_wallets & subsequent_participants) / initial_size * 100
            
            cohort_data.append({
                "cohort": f"{cohort_ido}\n({cohort_date.strftime('%Y-%m-%d')})",
                "ido_number": n,
                "retention": retention
            })
    
    return pd.DataFrame(cohort_data)

def create_participation_chart(participation_df):
    fig = go.Figure()
    
    pos_categories = ["new", "repeat", "reactivated"]
    colors_pos = ["#82ca9d", "#8884d8", "#ffc658"]
    
    x_labels = [f"{row['idoName']}<br>{row['date'].strftime('%Y-%m-%d')}" 
               for _, row in participation_df.iterrows()]
    
    for cat, color in zip(pos_categories, colors_pos):
        fig.add_trace(go.Bar(
            name=cat.capitalize(),
            x=x_labels,
            y=participation_df[cat],
            marker_color=color
        ))
    
    fig.add_trace(go.Bar(
        name="Churned",
        x=x_labels,
        y=[-v for v in participation_df["churned"]],
        marker_color="#ff7c43"
    ))
    
    fig.update_layout(
        barmode='relative',
        height=700,
        hovermode='x unified',
        yaxis_title="Number of Users",
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
        ),
        xaxis_title="IDO Projects",
        xaxis=dict(tickangle=0),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_sankey_data(raw_df):
    idos = raw_df.drop_duplicates('IDO Name', keep='first').sort_values(['Date', 'IDO_Order'])
    idos = idos.reset_index(drop=True)
    
    ido_labels = [f"{row['IDO Name']}\n({row['Date'].strftime('%Y-%m-%d')})" 
                 for _, row in idos.iterrows()]
    
    user_paths = defaultdict(list)
    all_wallets = set(raw_df['Wallet Address'])
    
    for i in range(len(idos)):
        current_ido = idos.loc[i, 'IDO Name']
        current_participants = set(raw_df[raw_df['IDO Name'] == current_ido]['Wallet Address'])
        
        previous_ido = None if i == 0 else idos.loc[i-1, 'IDO Name']
        previous_participants = set()
        if previous_ido:
            previous_participants = set(raw_df[raw_df['IDO Name'] == previous_ido]['Wallet Address'])
        
        all_historical_idos = set(idos.loc[:i-1, 'IDO Name']) if i > 0 else set()
        all_historical_participants = set()
        if all_historical_idos:
            all_historical_participants = set(raw_df[raw_df['IDO Name'].isin(all_historical_idos)]['Wallet Address'])
        
        for wallet in all_wallets:
            if wallet in current_participants:
                if i == 0:
                    state = 'New'
                elif wallet in previous_participants:
                    state = 'Repeat'
                elif wallet in all_historical_participants:
                    state = 'Reactivated'
                else:
                    state = 'New'
            else:
                if wallet in previous_participants:
                    state = 'Churned'
                elif wallet in all_historical_participants and wallet not in previous_participants:
                    state = 'Already Churned'
                else:
                    state = 'Never Participated'
            
            user_paths[wallet].append(state)
    
    nodes = []
    node_index = {}
    current_index = 0
    
    for i, ido in enumerate(ido_labels):
        states_in_ido = set(path[i] for path in user_paths.values() if len(path) > i)
        for state in sorted(states_in_ido):
            node_name = f"{state} ({ido})"
            nodes.append(node_name)
            node_index[node_name] = current_index
            current_index += 1
    
    sources = []
    targets = []
    values = []
    
    for i in range(len(ido_labels) - 1):
        flow_counts = defaultdict(int)
        
        for path in user_paths.values():
            if len(path) > i + 1:
                source_state = f"{path[i]} ({ido_labels[i]})"
                target_state = f"{path[i+1]} ({ido_labels[i+1]})"
                flow_counts[(source_state, target_state)] += 1
        
        for (source, target), value in flow_counts.items():
            if value > 0:
                sources.append(node_index[source])
                targets.append(node_index[target])
                values.append(value)
    
    return {
        'nodes': nodes,
        'sources': sources,
        'targets': targets,
        'values': values
    }

def display_sankey_diagram(sankey_data):
    # Define state order (top to bottom)
    state_order = ['Never Participated', 'New', 'Reactivated', 'Repeat', 'Churned', 'Already Churned']
    
    # Define base colors without opacity
    color_map = {
        "New": "#82ca9d",
        "Repeat": "#8884d8",
        "Reactivated": "#ffc658",
        "Churned": "#ff7c43",
        "Already Churned": "#ff7c43",
        "Never Participated": "rgb(128, 128, 128)"
    }
    
    # Helper function to convert hex/rgb to rgba or plain rgb
    def get_color(color, opacity=None):
        if color.startswith('#'):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        else:  # rgb format
            color = color.replace("rgb(", "").replace(")", "")
            r, g, b = map(int, color.split(","))
        
        if opacity is not None:
            return f"rgba({r}, {g}, {b}, {opacity})"
        return f"rgb({r}, {g}, {b})"
    
    # Helper function to get base color for a node
    def get_base_color(node_label):
        for state, color in color_map.items():
            if state in node_label:
                return color
        return color_map["Never Participated"]
    
    # Helper function to get node color
    def get_node_color(node_label, for_link=False):
        if "Never Participated" in node_label and not for_link:
            return get_color(color_map["Never Participated"], 0.1)  # 10% opacity for Never Participated nodes
        
        base_color = get_base_color(node_label)
        if for_link:
            return get_color(base_color, 0.25)  # 25% opacity for links
        return get_color(base_color)  # Full opacity for other nodes

    # Create node colors with special handling for Never Participated
    node_colors = [get_node_color(node) for node in sankey_data['nodes']]
    
    # Calculate source totals for percentage calculations
    source_totals = {}
    for source, value in zip(sankey_data['sources'], sankey_data['values']):
        source_totals[source] = source_totals.get(source, 0) + value

    # Create link colors and hover colors
    link_colors = []
    link_hover_colors = []
    hover_texts = []
    
    for source, target, value in zip(sankey_data['sources'], sankey_data['targets'], sankey_data['values']):
        source_node = sankey_data['nodes'][source]
        target_node = sankey_data['nodes'][target]
        
        # Get target node color for hover effect (full opacity)
        hover_color = get_node_color(target_node)
        
        # Set link color based on connection type and target color
        if "Never Participated" in source_node or "Never Participated" in target_node:
            link_color = "rgba(128, 128, 128, 0.10)"  # 10% opacity gray for Never Participated
        else:
            # Use target node's color with 25% opacity
            target_base_color = get_base_color(target_node)
            link_color = get_color(target_base_color, 0.25)
        
        # Calculate percentage
        percentage = (value / source_totals[source] * 100) if source_totals[source] > 0 else 0
        
        link_colors.append(link_color)
        link_hover_colors.append(hover_color)
        hover_texts.append(f"{int(value):,} participants ({percentage:.1f}%)")

    # Sort nodes within each column
    unique_idos = list(set([node.split(" (")[1] for node in sankey_data['nodes']]))
    sorted_indices = []
    current_index = 0
    
    for ido in unique_idos:
        # Get nodes for this IDO
        ido_nodes = [i for i, node in enumerate(sankey_data['nodes']) if ido in node]
        # Sort nodes based on state_order
        ido_nodes.sort(key=lambda x: [i for i, state in enumerate(state_order) 
                                    if state in sankey_data['nodes'][x]][0])
        sorted_indices.extend(ido_nodes)
        current_index += len(ido_nodes)

    # Create node line colors with special handling for Never Participated
    node_line_colors = []
    for node in sankey_data['nodes']:
        if "Never Participated" in node:
            node_line_colors.append("rgba(0, 0, 0, 0.1)")  # 10% opacity for Never Participated node borders
        else:
            node_line_colors.append("black")  # Full opacity for other node borders

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color=node_line_colors,
                width=0.5,
            ),
            label=sankey_data['nodes'],
            color=node_colors,
            hoverlabel=dict(
                font=dict(
                    size=13,
                    color='black'
                )
            )
        ),
        link=dict(
            source=sankey_data['sources'],
            target=sankey_data['targets'],
            value=sankey_data['values'],
            color=link_colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
            hoverlabel=dict(
                bgcolor=link_hover_colors,
                font=dict(
                    size=13,
                    color='black'
                )
            ),
            arrowlen=15
        )
    )])
    
    fig.update_layout(
        title_text="User Flow Between IDOs",
        height=1000,
        font=dict(
            size=13,
            color='black',
            family='Arial'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        template='none'
    )
    
    return fig


@st.cache_data(ttl=300)  # 300 seconds = 5 minutes
def load_data():
    SHEET_ID = "2PACX-1vST44Twi_xb-S5v-EhkqEgiIEX-9SevcqC0DHCHOcbwiIcP6k8LaZA_j5owb8D4r32r9vYJeaYlPZJa"
    url = f"https://docs.google.com/spreadsheets/d/e/{SHEET_ID}/pub?output=csv"
    
    try:
        raw_df = pd.read_csv(url)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='%d %b %Y')
        return raw_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def process_country_data(filtered_df):
    # Group by IDO Name, Date, and Country, count unique wallet addresses
    country_data = filtered_df.groupby(['IDO Name', 'Date', 'Country'])['Wallet Address'].nunique().reset_index()
    country_data.columns = ['IDO Name', 'Date', 'Country', 'Participants']
    
    # Calculate the total participants for each IDO
    ido_totals = country_data.groupby(['IDO Name', 'Date'])['Participants'].sum().reset_index()
    ido_totals.columns = ['IDO Name', 'Date', 'Total Participants']
    
    # Merge the totals back to the country_data
    country_data = country_data.merge(ido_totals, on=['IDO Name', 'Date'])
    
    # Calculate the percentage
    country_data['Percentage'] = country_data['Participants'] / country_data['Total Participants'] * 100
    
    # Sort by IDO Name, Date, and Percentage
    country_data = country_data.sort_values(['IDO Name', 'Date', 'Percentage'], ascending=[True, True, False])
    
    # Keep only top 20 countries for each IDO
    country_data = country_data.groupby('IDO Name').apply(lambda x: x.nlargest(20, 'Participants')).reset_index(drop=True)
    
    # Get the IDO order from the filtered_df
    ido_order = filtered_df.drop_duplicates('IDO Name').sort_values('Date')['IDO Name'].tolist()
    
    # Add the IDO order to the country_data
    country_data['IDO_Order'] = country_data['IDO Name'].map({ido: i for i, ido in enumerate(ido_order)})
    
    # Sort the country_data by IDO_Order and Date
    country_data = country_data.sort_values(['IDO_Order', 'Date'])
    
    return country_data

def process_country_data_for_bar_race(country_data):
    # Pivot the data to create a matrix of countries vs IDOs
    pivot_data = country_data.pivot(index=['IDO Name', 'Date'], columns='Country', values='Participants').fillna(0)
    
    # Sort IDOs by date
    pivot_data = pivot_data.sort_index(level='Date')
    
    # Get top 20 countries by total participation
    top_20_countries = pivot_data.sum().nlargest(20).index.tolist()
    
    # Filter for top 20 countries
    pivot_data = pivot_data[top_20_countries]
    
    # Calculate cumulative maximum for each country
    cumulative_max = pivot_data.cummax()
    
    # Reset index to make IDO Name the only index
    pivot_data = pivot_data.reset_index(level='Date')
    cumulative_max = cumulative_max.reset_index(level='Date')
    
    return pivot_data, cumulative_max

def create_animated_bar_chart(pivot_data, cumulative_max):
    countries = pivot_data.columns[1:]  # Exclude 'Date' column
    
    # Calculate median participation for each country
    median_participation = pivot_data.iloc[:, 1:].median()
    
    # Create a custom color palette for 20 countries
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    colors = colors[:len(countries)]  # Ensure we have exactly as many colors as countries
    
    # Create the initial figure without frames and sliders
    fig = go.Figure(
        data=[
            # Historical maximum "ghost" bars
            go.Bar(
                x=countries,
                y=cumulative_max.iloc[0, 1:],
                marker=dict(color='rgba(200,200,200,0.25)'),
                name='Historical Maximum'
            ),
            # Current participation bars
            go.Bar(
                x=countries,
                y=pivot_data.iloc[0, 1:],
                text=pivot_data.iloc[0, 1:],
                texttemplate='%{text:,.0f}',
                textposition='outside',
                textfont=dict(size=10),
                marker=dict(color=colors),
                name='Current Participation'
            ),
            # Median participation as error bars
            go.Bar(
                x=countries,
                y=median_participation,  # Use median as the base
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[0] * len(countries),  # No upper error
                    arrayminus=median_participation,  # Lower error goes down to 0
                    visible=True,
                    color='rgba(0,0,0,0.6)',
                    thickness=1.5,
                    width=6
                ),
                marker=dict(color='rgba(0,0,0,0)', line=dict(width=0)),  # Invisible bar
                name='Median Participation'
            )
        ],
        layout=go.Layout(
            title="Top 20 Countries Participation by IDO",
            xaxis=dict(tickangle=-45, title=None),
            yaxis=dict(title="Number of Participants", range=[0, 40]),  # Set y-axis range
            barmode='overlay'
        )
    )
    
    # Create frames
    frames = [
        go.Frame(
            data=[
                # Historical maximum "ghost" bars
                go.Bar(x=countries,
                       y=cumulative_max.iloc[i, 1:],
                       marker=dict(color='rgba(200,200,200,0.25)')),
                # Current participation bars
                go.Bar(x=countries,
                       y=pivot_data.iloc[i, 1:],
                       text=pivot_data.iloc[i, 1:],
                       marker=dict(color=colors)),
                # Median participation as error bars (stays constant)
                go.Bar(
                    x=countries,
                    y=median_participation,  # Use median as the base
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[0] * len(countries),  # No upper error
                        arrayminus=median_participation,  # Lower error goes down to 0
                        visible=True,
                        color='rgba(0,0,0,0.6)',
                        thickness=1.5,
                        width=6
                    ),
                    marker=dict(color='rgba(0,0,0,0)', line=dict(width=0)),  # Invisible bar
                    name='Median Participation'
                )
            ],
            name=f"{pivot_data.index[i]} ({pivot_data.iloc[i]['Date'].strftime('%Y-%m-%d')})",
            layout=go.Layout(
                title_text=f"Top 20 Countries Participation: {pivot_data.index[i]} ({pivot_data.iloc[i]['Date'].strftime('%Y-%m-%d')})",
                yaxis=dict(range=[0, 40])  # Set y-axis range for each frame
            )
        )
        for i in range(len(pivot_data))
    ]
    
    # Add frames to the figure
    fig.frames = frames
    
    # Create slider
    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16},
            "prefix": "IDO: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [
                    [f.name],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300, "easing": "quadratic-in-out"}
                    }
                ],
                "label": f"{pivot_data.index[k]} ({pivot_data.iloc[k]['Date'].strftime('%Y-%m-%d')})",
                "method": "animate"
            }
            for k, f in enumerate(frames)
        ]
    }]
    
    # Add slider to the figure
    fig.update_layout(
        sliders=sliders,
        height=600,
        legend_title_text='Participation',
        yaxis=dict(range=[0, 40])  # Ensure y-axis range is set in the final layout
    )
    
    return fig

# Load initial data
raw_df = load_data()

if not raw_df.empty:
    # Add filters in a container
    with st.container():
        # Project type filter
        project_type = st.radio(
            "Select Project Type:",
            ["All Projects", "Node Projects Only", "IDO Projects Only"],
            horizontal=True
        )
        
        # Eclipse Fi filters (checked by default)
        exclude_eclipse_presale = st.checkbox("Exclude Eclipse Fi pre-sale", value=True)
        exclude_eclipse = st.checkbox("Exclude Eclipse Fi", value=True)
        
        st.divider()  # Add a visual separator
    
    # Apply filters to create filtered_df
    filtered_df = raw_df.copy()
    
    # Apply project type filter
    if project_type == "Node Projects Only":
        filtered_df = filtered_df[filtered_df['Type'] == 'Node']
    elif project_type == "IDO Projects Only":
        filtered_df = filtered_df[filtered_df['Type'] == 'IDO']
    
    # Apply Eclipse Fi filters
    projects_to_exclude = []
    if exclude_eclipse_presale:
        projects_to_exclude.append("Eclipse Fi pre-sale")
    if exclude_eclipse:
        projects_to_exclude.append("Eclipse Fi")
    
    if projects_to_exclude:
        filtered_df = filtered_df[~filtered_df['IDO Name'].isin(projects_to_exclude)]
    
    # Sort and create IDO order
    ido_dates = filtered_df.sort_values('Date').groupby('IDO Name')['Date'].first()
    ido_order = {ido: idx for idx, ido in enumerate(ido_dates.index)}
    filtered_df['IDO_Order'] = filtered_df['IDO Name'].map(ido_order)
    filtered_df = filtered_df.sort_values(['Date', 'IDO_Order'])
    
    # Process data for visualizations using filtered_df
    participation_df = process_participation_data(filtered_df)
    cohort_df = process_cohort_data(filtered_df)

    # Process country data
    country_data = process_country_data(filtered_df)



    # Create two columns for the first row of charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cohort Retention by IDO Participation")
        
        fig_retention = px.line(cohort_df, 
                            x="ido_number", 
                            y="retention", 
                            color="cohort",
                            title="Retention by Cohort",
                            labels={"ido_number": "Nth IDO Participation",
                                    "retention": "Retention Rate (%)",
                                    "cohort": "Started with"})
        
        fig_retention.update_layout(
            height=700,
            hovermode='x unified',
            xaxis=dict(
                dtick=1,
                tick0=1
            )
        )
        st.plotly_chart(fig_retention, use_container_width=True)
        
        st.caption("""Shows how each cohort (grouped by their first IDO) continues to participate in subsequent IDOs.
                    Each line represents users who started with a specific IDO.""")

    with col2:
        st.subheader("User Participation Breakdown by IDO")
        
        fig_participation = create_participation_chart(participation_df)
        fig_participation.update_layout(height=700)
        st.plotly_chart(fig_participation, use_container_width=True)
        
        st.caption("""
        - New: First-time participants
        - Repeat: Participated in the previous IDO
        - Reactivated: Returning after skipping at least one IDO (participated in any previous IDO)
        - Churned: Users who didn't participate after previous participation (shown as negative)
        """)

    # Add Sankey diagram in full width below
    st.subheader("User Flow Analysis")
    sankey_data = create_sankey_data(filtered_df)
    fig_sankey = display_sankey_diagram(sankey_data)
    fig_sankey.update_layout(height=1000)
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    st.caption("""
    This Sankey diagram shows how users flow between different states across IDOs:
    - Green: New users in each IDO
    - Purple: Repeat users from the previous IDO
    - Yellow: Reactivated users who skipped at least one IDO
    - Orange: Users who churned (didn't participate in this IDO)
    The width of each flow indicates the number of users taking that path.
    """)

    # Add animated bar chart race at the bottom
    st.subheader("Top 20 Countries Participation by IDO")

    # Process data for animated bar chart race
    pivot_data, cumulative_max = process_country_data_for_bar_race(country_data)
    fig_bar_race = create_animated_bar_chart(pivot_data, cumulative_max)

    st.plotly_chart(fig_bar_race, use_container_width=True, key="country_bar_race")

    st.caption("""
    This animated bar chart shows how the top 20 countries' participation changes across IDOs.
    Each frame represents an IDO, sorted by their date of occurrence from earliest to latest.
    The colored bars represent the number of participants from each country for the current IDO.
    The light gray "ghost" bars show the highest number of participants achieved by each country up to that point.
    The black error bars indicate the median participation 'water level' for each country across all IDOs.
    Use the slider to move between IDOs and watch the bars animate smoothly.
    """)

    # Add data processing section
    st.subheader("Data Processing")
    with st.expander("Show Raw Data"):
        st.write("Raw Data (sorted by date)")
        st.dataframe(filtered_df.sort_values('Date'))
        st.write("Processed Participation Data")
        st.dataframe(participation_df)
        st.write("Processed Cohort Data")
        st.dataframe(cohort_df)
else:
    st.error("No data available. Please check the Google Sheets URL and permissions.")










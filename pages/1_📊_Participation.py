import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from bson import ObjectId
import requests
import base64
import json
from datetime import datetime, timezone
import numpy as np
import time
from requests.exceptions import RequestException
import io
from decimal import Decimal

# Add this near the top of your script, after the imports
st.set_page_config(layout="wide")

# Set up MongoDB connection
@st.cache_resource
def init_connection():
    return MongoClient(st.secrets["mongo"]["connection_string"])

client = init_connection()

# Function to get projects from MongoDB
@st.cache_data
def get_projects():
    db = client.IDO
    projects = list(db.production.find({}, {
        "id": 1, 
        "info.name": 1, 
        "token": 1,  # Include the entire token object
        "contracts": 1
    }))
    return projects

# Whitelist applicants only need to be loaded once per session
@st.cache_data
def get_whitelist_applicants(project_id):
    db = client.IDO
    applicants = list(db.whitelist.find({"project_id": project_id}))
    return applicants

# Keep ideal allocations updating every minute
@st.cache_data
def get_ideal_allocations(project_id):
    db = client.IDO
    applicants = list(db.whitelist.find({"project_id": project_id}, {"wallet_address": 1, "form_data.idealAllocation": 1}))
    return applicants

# Keep smart contract queries updating every minute
@st.cache_data
def query_smart_contract(contract_address, query):
    rest_endpoint = st.secrets["neutron"]["rpc_url"]
    encoded_query = base64.b64encode(json.dumps(query).encode()).decode()
    
    url = f"{rest_endpoint}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{encoded_query}"

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if "data" in result:
                return result
            else:
                raise ValueError("Unexpected response format")
        
        except (RequestException, ValueError) as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                st.error("Max retries reached. Unable to query smart contract.")
                st.error(f"REST Endpoint: {rest_endpoint}")
                st.error(f"Error details: {str(e)}")
                return None

# Function to calculate tier based on essence
def calculate_tier(essence):
    if essence < 1500:
        return "No Tier"
    elif essence < 4000:
        return "Star Dust"
    elif essence < 5000:
        return "Comet"
    elif essence < 20000:
        return "Solar System"
    elif essence < 40000:
        return "Constellation"
    elif essence < 80000:
        return "Galaxy"
    else:
        return "Multiverse"

# Function to get contract address
def get_contract_address(project):
    if "contracts" in project and "presale" in project["contracts"]:
        return project["contracts"]["presale"]
    else:
        st.warning(f"Presale contract address not found for project {project['info']['name']}. Please enter it manually.")
        return st.text_input("Enter presale contract address manually:")

# Function to define tier order
def tier_order(tier):
    tier_ranks = {
        "No Tier": 0,
        "Star Dust": 1,
        "Comet": 2,
        "Solar System": 3,
        "Constellation": 4,
        "Galaxy": 5,
        "Multiverse": 6
    }
    return tier_ranks.get(tier, -1)  # Return -1 for unknown tiers

def get_essence_csv(project_id):
    base_url = "https://raw.githubusercontent.com/cryptomius/eclipsefi-utils/main/data/cosmic-essence-snapshots/"
    file_name = f"{project_id.lower()}.csv"
    url = f"{base_url}{file_name}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return io.StringIO(response.text)
    else:
        st.error(f"Failed to fetch CSV for project {project_id}. Status code: {response.status_code}")
        return None

# Add this function to check if FCFS is still active
@st.cache_data
def is_fcfs_active(project_id):
    db = client.IDO
    project = db.production.find_one({"id": project_id}, {"token.fcfs_ido_end": 1})
    if not project or "token" not in project or "fcfs_ido_end" not in project["token"]:
        return False
    
    end_time = datetime.fromisoformat(project["token"]["fcfs_ido_end"].replace('Z', '+00:00'))
    return datetime.now(end_time.tzinfo) < end_time

# Add this new function after query_smart_contract
@st.cache_data
def get_target_amount(contract_address):
    try:
        fund_config = query_smart_contract(contract_address, {"query_fund_config": {}})
        
        if fund_config and "data" in fund_config:
            total_rewards = Decimal(fund_config["data"]["total_rewards_amount"])
            exchange_rate = Decimal(fund_config["data"]["exchange_rate"])
            target = float(total_rewards / exchange_rate / Decimal('1000000'))
            return target
        
        return None
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
        st.warning(f"Error calculating target amount from smart contract: {str(e)}")
        return None

# Main app
def main():
    st.title("Eclipse Fi IDO Analytics Dashboard")

    # Remove snapshot_time_container since we don't need it anymore
    status_text_container = st.empty()

    # Add refresh button and last updated time in columns
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh Data"):
            # Clear caches for functions that need refreshing
            query_smart_contract.clear()
            get_ideal_allocations.clear()
            is_fcfs_active.clear()
            st.rerun()
    with col2:
        current_time = datetime.now(timezone.utc)
        st.markdown(f"*Last updated: {current_time.strftime('%Y-%m-%d %H:%M UTC')}*")

    # Get projects and create selectbox
    projects = get_projects()
    
    # Filter out projects with "Eclipse Fi" in the name and those without fcfs_ido_end
    projects = [p for p in projects if 
                "Eclipse Fi" not in p["info"]["name"] and 
                p.get("token", {}).get("fcfs_ido_end") is not None]
    
    # Sort projects, putting None dates at the end
    projects.sort(
        key=lambda x: (
            not bool(x.get("token", {}).get("whitelist_ido_start")),  # Empty/None dates go last
            x.get("token", {}).get("whitelist_ido_start") or "9999-12-31"  # Then sort by date
        ), 
        reverse=False
    )
    
    # Separate projects with dates and without dates
    projects_with_dates = [p for p in projects if p.get("token", {}).get("whitelist_ido_start")]
    projects_without_dates = [p for p in projects if not p.get("token", {}).get("whitelist_ido_start")]
    
    # Reverse only the projects with dates
    projects_with_dates.reverse()
    
    # Combine the lists back together
    projects = projects_with_dates + projects_without_dates
    
    # Create project names with dates and find default selection
    project_names = []
    default_index = 0
    found_valid_csv = False
    
    for i, p in enumerate(projects):
        name = p["info"]["name"]
        start_date = p.get("token", {}).get("whitelist_ido_start", "")
        if start_date:
            try:
                date_obj = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                formatted_date = date_obj.strftime("%d %b %Y")
                name = f"{name} ({formatted_date})"
            except ValueError:
                pass
        project_names.append(name)
        
        # Only check for CSV if we haven't found a valid one yet
        if not found_valid_csv:
            csv_content = get_essence_csv(p["id"])
            if csv_content is not None:
                default_index = i
                found_valid_csv = True
    
    selected_project_name = st.selectbox("Select a project", project_names, index=default_index)
    
    # Extract the original project name without the date for lookup
    selected_project_name_only = selected_project_name.split(" (")[0]
    selected_project = next(p for p in projects if p["info"]["name"] == selected_project_name_only)
    project_id = selected_project["id"]
    
    # Handle the total_raised value safely
    try:
        total_raised_str = selected_project["token"]["total_raised"]
        if total_raised_str.lower() == 'tba':
            total_raised = 0
        else:
            # Strip out any non-numeric characters except decimal points
            numeric_str = ''.join(c for c in total_raised_str if c.isdigit() or c == '.')
            total_raised = float(numeric_str)
    except (ValueError, KeyError, AttributeError):
        total_raised = 0
        st.warning("Total raised amount not available or invalid")

    # Get contract address
    contract_address = get_contract_address(selected_project)

    # Fetch CSV from GitHub
    csv_content = get_essence_csv(project_id)
    
    if csv_content is not None and contract_address:
        try:
            essence_df = pd.read_csv(csv_content)
            
            # Check if 'wallets' and 'essence' columns exist (case-insensitive)
            required_columns = {'wallets': 'wallet', 'essence': 'essence'}
            missing_columns = []
            
            for col, rename in required_columns.items():
                if col.lower() not in [c.lower() for c in essence_df.columns]:
                    missing_columns.append(col)
                else:
                    # Rename the column to ensure consistent naming
                    essence_df.rename(columns={c: rename for c in essence_df.columns if c.lower() == col.lower()}, inplace=True)
            
            if missing_columns:
                st.error(f"The CSV is missing the following required columns: {', '.join(missing_columns)}")
                st.error("Please ensure the CSV file has 'wallets' and 'essence' columns.")
                return
            
            essence_df["essence"] = essence_df["essence"] / 1000000
            essence_df["tier"] = essence_df["essence"].apply(calculate_tier)

            # Get whitelist applicants
            applicants = get_whitelist_applicants(project_id)
            applicant_wallets = [a["wallet_address"] for a in applicants]

            # Query smart contract for allocation list
            allocation_list = query_smart_contract(contract_address, {"query_allocation_list": {}})
            if allocation_list is None:
                st.error("Failed to retrieve allocation list. Please check your network connection and try again.")
                return

            winner_wallets = [w["neutron_address"] for w in allocation_list["data"]]

            # Query smart contract for participant list
            participant_list = query_smart_contract(contract_address, {"query_participant_list": {}})
            if participant_list is None:
                st.error("Failed to retrieve participant list. Please check your network connection and try again.")
                return
            
            # Process participant data
            participant_data = []
            for p in participant_list["data"]:
                wallet = p["address"]
                # Use Decimal for precise calculations
                funded_private = Decimal(p["participant"]["funded_private"]) / Decimal('1000000')
                funded_public = Decimal(p["participant"]["funded_public"]) / Decimal('1000000')
                participant_data.append({
                    "wallet": wallet,
                    "funded_private": float(funded_private),  # Convert to float for DataFrame compatibility
                    "funded_public": float(funded_public),
                    "total_funded": float(funded_private + funded_public)
                })
            
            participant_df = pd.DataFrame(participant_data)

            # Calculate total raised using sum of original Decimal values to maintain precision
            total_raised_actual = sum(
                (Decimal(p["participant"]["funded_private"]) + Decimal(p["participant"]["funded_public"])) 
                for p in participant_list["data"]
            ) / Decimal('1000000')

            # Merge data
            merged_df = essence_df.merge(pd.DataFrame({"wallet": applicant_wallets}), on="wallet", how="inner")
            merged_df["is_winner"] = merged_df["wallet"].isin(winner_wallets)
            merged_df = merged_df.merge(participant_df, on="wallet", how="left")
            merged_df["total_funded"] = merged_df["total_funded"].fillna(0)

            # Visualizations
            st.header("Project Overview")
            
            # Get the end time and format current time
            db = client.IDO
            project = db.production.find_one({"id": project_id}, {"token.fcfs_ido_end": 1})
            
            # Use datetime.now(UTC) for current time
            current_time = datetime.now(timezone.utc)
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M UTC")
            
            if project and "token" in project and "fcfs_ido_end" in project["token"]:
                end_time = datetime.fromisoformat(project["token"]["fcfs_ido_end"].replace('Z', '+00:00'))
                
                # Format the countdown or show IDO closed
                if current_time > end_time:
                    status_text = "IDO closed"
                else:
                    time_left = end_time - current_time
                    days = time_left.days
                    hours = time_left.seconds // 3600
                    minutes = (time_left.seconds % 3600) // 60
                    
                    parts = []
                    if days > 0:
                        parts.append(f"{days} days")
                    if hours > 0:
                        parts.append(f"{hours} hours")
                    if minutes > 0:
                        parts.append(f"{minutes} minutes")
                    
                    status_text = ", ".join(parts) + " remaining"
                
                # Only update the status text
                status_text_container.markdown(f"**{status_text}**")
            else:
                # If no end time found, just show snapshot time
                status_text_container.markdown(f"*Snapshot as of: {current_time_str}*")

            col1, col2 = st.columns(2)
            with col1:
                # Get target amount from smart contract or fall back to MongoDB value
                target_amount = get_target_amount(contract_address)
                if target_amount is None:
                    target_amount = total_raised  # Fall back to MongoDB value
                
                # Calculate totals for each phase
                total_private = participant_df["funded_private"].sum()
                total_public = participant_df["funded_public"].sum()
                
                # Create the gauge without the annotation
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=float(total_raised_actual),
                    title={'text': "Total Raised (USDC)"},
                    delta={
                        'reference': target_amount,
                        'decreasing': {'color': "gray"},
                        'increasing': {'color': "gray"},
                        'position': "bottom"
                    },
                    number={
                        'valueformat': '.3s',
                        'font': {'color': "darkblue"},
                        'suffix': ''
                    },
                    gauge={
                        'axis': {'range': [None, target_amount]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, float(total_private)], 'color': "rgb(200, 200, 255)", 'name': "Whitelist Sale"},
                            {'range': [float(total_private), float(total_raised_actual)], 'color': "rgb(100, 100, 255)", 'name': "FCFS"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': target_amount
                        }
                    }
                ))

                # Update the layout height from 300 to 390 (30% increase)
                fig.update_layout(height=390)
                
                # Display the gauge
                st.plotly_chart(fig)
                
                # Display the breakdown text below the gauge
                st.markdown(f"""
                    - Whitelist Sale: {total_private:,.2f} USDC
                    - FCFS: {total_public:,.2f} USDC
                    - Target: {target_amount:,.2f} USDC
                """)

            with col2:
                tier_counts = merged_df["tier"].value_counts().reset_index()
                tier_counts.columns = ["Tier", "Count"]
                tier_counts["Type"] = "Applicants"
                
                winner_tier_counts = merged_df[merged_df["is_winner"]]["tier"].value_counts().reset_index()
                winner_tier_counts.columns = ["Tier", "Count"]
                winner_tier_counts["Type"] = "Winners"
                
                tier_comparison = pd.concat([tier_counts, winner_tier_counts])
                tier_comparison["TierOrder"] = tier_comparison["Tier"].apply(tier_order)
                tier_comparison = tier_comparison.sort_values("TierOrder")
                
                fig = px.bar(tier_comparison, x="Tier", y="Count", color="Type", barmode="group",
                             title="Tier Breakdown: Applicants vs Winners",
                             category_orders={"Tier": sorted(tier_comparison["Tier"].unique(), key=tier_order)})
                st.plotly_chart(fig)

            st.header("Contribution Analysis")
            col1, col2 = st.columns(2)
            with col1:
                tier_contribution = merged_df.groupby("tier").agg({
                    "funded_private": "sum",
                    "funded_public": "sum"
                }).reset_index()
                tier_contribution = tier_contribution.melt(id_vars=["tier"], 
                                                           var_name="Phase", 
                                                           value_name="Amount")
                # Update the Phase names
                tier_contribution["Phase"] = tier_contribution["Phase"].replace({
                    "funded_private": "Whitelist Sale",
                    "funded_public": "FCFS"
                })
                tier_contribution["TierOrder"] = tier_contribution["tier"].apply(tier_order)
                tier_contribution = tier_contribution.sort_values("TierOrder")
                
                fig = px.bar(tier_contribution, x="tier", y="Amount", color="Phase", 
                             title="USDC Committed by Tier and Phase",
                             category_orders={"tier": sorted(tier_contribution["tier"].unique(), key=tier_order)})
                st.plotly_chart(fig)

            with col2:
                contributor_counts = merged_df[merged_df["total_funded"] > 0].groupby("tier").size().reset_index(name="count")
                contributor_counts["TierOrder"] = contributor_counts["tier"].apply(tier_order)
                contributor_counts = contributor_counts.sort_values("TierOrder")
                
                fig = px.bar(contributor_counts, x="tier", y="count", 
                             title="Number of Contributors by Tier",
                             category_orders={"tier": sorted(contributor_counts["tier"].unique(), key=tier_order)})
                st.plotly_chart(fig)

            st.header("Ideal vs Actual Allocations by Tier")

            # Get ideal allocations
            ideal_allocations = get_ideal_allocations(project_id)

            # Create DataFrame with ideal allocations
            ideal_df = pd.DataFrame([(a['wallet_address'], a['form_data'].get('idealAllocation', '0')) 
                                     for a in ideal_allocations], 
                                    columns=['wallet', 'ideal_allocation'])

            # Clean ideal allocation values
            ideal_df['ideal_allocation'] = ideal_df['ideal_allocation'].apply(lambda x: float(''.join(filter(str.isdigit, str(x)))) if x else 0)

            # Merge with essence data to get tiers
            ideal_df = ideal_df.merge(essence_df[['wallet', 'tier']], on='wallet', how='left')

            # Group by tier and sum ideal allocations
            tier_allocations = ideal_df.groupby('tier')['ideal_allocation'].sum().reset_index()

            # Get actual allocations from the USDC Committed by Tier and Phase data
            actual_allocations = merged_df.groupby('tier').agg({
                'funded_private': 'sum',
                'funded_public': 'sum'
            }).reset_index()
            actual_allocations['total_funded'] = actual_allocations['funded_private'] + actual_allocations['funded_public']

            # Merge ideal and actual allocations
            combined_allocations = tier_allocations.merge(actual_allocations[['tier', 'total_funded']], on='tier', how='outer').fillna(0)
            combined_allocations['TierOrder'] = combined_allocations['tier'].apply(tier_order)
            combined_allocations = combined_allocations.sort_values('TierOrder')

            # Create the enhanced bar chart
            fig = go.Figure()

            # Add ideal allocations as ghost background
            fig.add_trace(go.Bar(
                x=combined_allocations['tier'],
                y=combined_allocations['ideal_allocation'],
                name='Ideal Allocation',
                marker_color='rgba(200, 200, 200, 0.5)'  # Light gray, semi-transparent
            ))

            # Add actual allocations
            fig.add_trace(go.Bar(
                x=combined_allocations['tier'],
                y=combined_allocations['total_funded'],
                name='Actual Allocation',
                marker_color='rgba(0, 123, 255, 0.8)'  # Blue, slightly transparent
            ))

            # Update layout
            fig.update_layout(
                title='Ideal vs Actual Allocations by Tier',
                xaxis_title='Tier',
                yaxis_title='Allocation Amount',
                barmode='overlay',
                xaxis={'categoryorder': 'array', 'categoryarray': sorted(combined_allocations['tier'].unique(), key=tier_order)}
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")
            st.error("Please check your CSV file format and try again.")
            return

    else:
        if not contract_address:
            st.error("Please provide a valid contract address to proceed.")
        if csv_content is None:
            st.warning("Failed to fetch the CSV file for the selected project.")

if __name__ == "__main__":
    main()

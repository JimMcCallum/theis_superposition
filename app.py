import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expi, jn
from scipy.optimize import minimize
from matplotlib.patches import Polygon as MPLPolygon

# ============================================================================
# BENCH PROGRESSION DATA FOR MINE DEWATERING
# ============================================================================

# North Pit bench progression (time in years, elevation in meters)
NORTH_PIT_BENCHES = np.array([
    [0, 678],
    [0.5, 668],
    [1, 668],
    [1.4, 658],
    [1.8, 658],
    [2.1, 648],
    [2.4, 648],
    [2.7, 638],
    [3, 638],
    [3.2, 628],
    [3.4, 628],
    [3.4, 618],
    [3.6, 618],
    [3.8, 608],
    [4, 608],
    [4.2, 598],
    [4.4, 598],
    [4.6, 588],
    [15, 588]
])

# South Pit bench progression (time in years, elevation in meters)
SOUTH_PIT_BENCHES = np.array([
    [0, 678],
    [0.5, 668],
    [1, 668],
    [1.4, 658],
    [1.8, 658],
    [2.1, 648],
    [2.4, 648],
    [2.7, 638],
    [3, 638],
    [3.2, 628],
    [10.2, 628],
    [10.4, 618],
    [10.6, 618],
    [10.8, 608],
    [11, 608],
    [16, 608]
])

# Initial water table elevation (meters above datum)
INITIAL_WATER_TABLE = 647


# ============================================================================
# WELL MODEL CLASSES
# ============================================================================

class Theis_well:
    """Theis solution for confined aquifer"""
    def __init__(self, x, y, Q, ton, T, S, rwell, toff = None):
        self.x = x
        self.y = y
        self.T = T
        self.S = S
        self.rwell = rwell
        self.Q = Q
        self.ton = ton
        self.toff = toff
        
    def s(self, X, Y, t):
        r = np.sqrt((self.x - X)**2. + (self.y - Y)**2.)
        if np.isscalar(r):
            if r < self.rwell:
                r = self.rwell
        else:
            r[r < self.rwell] = self.rwell
        u = r**2 * self.S / (4. * (t - self.ton) * self.T)
        Wu = -expi(-u)
        sdum = self.Q / (4 * np.pi * self.T) * Wu
        if np.isscalar(t):
            if t <= self.ton:
                sdum = 0.
        else:
            sdum[t <= self.ton] = 0.
        if self.toff is not None:
            u2 = r**2 * self.S / (4. * (t - self.toff) * self.T)
            Wu2 = -expi(-u2)
            sdum2 = self.Q / (4 * np.pi * self.T) * Wu2
            if np.isscalar(t):
                if t <= self.toff:
                    sdum2 = 0
            else:
                sdum2[t <= self.toff] = 0
            sdum -= sdum2
        return sdum



import numpy as np
from scipy.optimize import brentq
from scipy.special import jn

def newman_roots0_func(gamma, y, sigma):
    return sigma*gamma*np.sinh(gamma) - (y**2 - gamma**2)*np.cosh(gamma)

def newman_roots_func(gamma, y, sigma):
    return sigma*gamma*np.sin(gamma) + (y**2 + gamma**2)*np.cos(gamma)


class Neuman_well:
    """Neuman solution for unconfined aquifer with delayed yield"""
    def __init__(self, x, y, Q, ton, T, S, b, anis, Sy, rwell=0.1, dD=0., lD=1.):
        self.x = x
        self.y = y
        self.Q = Q
        self.ton = ton
        self.T = T
        self.S = S
        self.b = b
        self.anis = anis  # Kz/Kr
        self.Sy = Sy
        self.dD = dD
        self.lD = lD
        self.rwell = rwell
        
    def s(self, X, Y, t, z1D=0.01, z2D=0.99):
        r = np.sqrt((self.x - X)**2 + (self.y - Y)**2)
        if np.isscalar(r):
            if r < self.rwell:
                r = self.rwell
        else:
            r[r < self.rwell] = self.rwell

        Beta = r**2 * self.anis / (self.b**2)
        sigma = self.S / self.Sy
        ts = self.T * t / (self.S * r**2)

        dy = 0.1
        y = 0.0
        idum2 = 0.0
        barry = True
        
        # Tuned tolerances for faster convergence
        xtol = 1e-10  # Relaxed from default 2e-12
        rtol = 1e-10  # Relaxed from default ~8.9e-16
        maxiter = 50  # Reduced from default 100
        
        while barry:
            y += dy

            # --- Hyperbolic root in [-y, y] ---
            try:
                if newman_roots0_func(0, y, sigma) * newman_roots0_func(y, y, sigma) < 0:
                    gamma0 = brentq(newman_roots0_func, 0, y, args=(y, sigma),
                                   xtol=xtol, rtol=rtol, maxiter=maxiter)
                else:
                    gamma0 = None
            except ValueError:
                gamma0 = None

            idum1 = 0.0
            if gamma0 is not None:
                u = 1. - np.exp(-ts*Beta*(y**2 - gamma0**2))
                #u *= (np.sinh(gamma0*z2D) - np.sinh(gamma0*z1D))
                u*= np.tanh(gamma0)
                u /= (y**2 + (1+sigma)*gamma0**2 - (y**2 - gamma0**2)**2/sigma) * gamma0
                #u *= np.sinh(gamma0*(1-self.dD)) - np.sinh(gamma0*(1-self.lD))
                #u /= (z2D-z1D)*(self.lD - self.dD)*np.sinh(gamma0)
                idum1 += u

            # --- Oscillatory roots in [(2j-1)π/2, jπ] ---
            j = 0
            while True:
                j += 1
                a, b = ( (2*j-1)*np.pi/2, j*np.pi )
                try:
                    if newman_roots_func(a, y, sigma) * newman_roots_func(b, y, sigma) < 0:
                        gamma = brentq(newman_roots_func, a, b, args=(y, sigma),
                                      xtol=xtol, rtol=rtol, maxiter=maxiter)
                    else:
                        break
                except ValueError:
                    break

                u = 1. - np.exp(-ts*Beta*(y**2 + gamma**2))
                #u *= (np.sin(gamma*z2D) - np.sin(gamma*z1D))
                u *= np.tan(gamma)
                u /= (y**2 - (1+sigma)*gamma**2 - (y**2 + gamma**2)**2/sigma) * gamma
                #u *= np.sin(gamma*(1-self.dD)) - np.sin(gamma*(1-self.lD))
                #u /= (z2D-z1D)*(self.lD - self.dD)*np.sin(gamma)

                idum1 += u
                if abs(u) < 1e-7:
                    break

            dit = 4*y*jn(0, y*np.sqrt(Beta)) * idum1 * dy
            idum2 += dit
            if abs(dit) < 1e-7:
                barry = False

        return self.Q/(4*np.pi*self.T) * idum2


# ============================================================================
# UTILITY FUNCTIONS FOR POLYGON OPERATIONS
# ============================================================================

def point_in_polygon(x, y, polygon_points):
    """Check if point (x,y) is inside polygon using ray casting algorithm"""
    n = len(polygon_points)
    inside = False
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(layout="wide", page_title="Well Testing Simulator")

# Sidebar for session info
with st.sidebar:
    st.header("📋 Session Info")
    if 'student_id' in st.session_state:
        st.metric("Session ID", st.session_state.student_id)
        
        st.markdown("---")
        st.caption(f"Wells placed: {sum(len(st.session_state[k]) for k in ['wells_Pumping', 'wells_Injection', 'wells_Monitoring'])}")
        st.caption(f"Areas defined: {len(st.session_state.get('polygons', []))}")
        
        # Show current fitted parameters if available
        if 'fitted_params' in st.session_state and isinstance(st.session_state.fitted_params, dict) and st.session_state.fitted_params:
            st.markdown("---")
            st.subheader("🎯 Fitted Parameters")
            
            try:
                for pump_well, fp in st.session_state.fitted_params.items():
                    # Check if fp is actually a dictionary (not a string or other type)
                    if isinstance(fp, dict):
                        with st.expander(f"**{pump_well}**", expanded=True):
                            st.caption(f"**Monitoring:** {fp.get('monitoring_well', 'N/A')}")
                            st.caption(f"**T:** {fp.get('T', 0):.1f} m²/day")
                            st.caption(f"**S:** {fp.get('S', 0):.2e}")
                            st.caption(f"**Q:** {fp.get('Q', 0):.1f} m³/day")
                            st.caption(f"**RMSE:** {fp.get('rmse', 0):.4f} m")
            except Exception as e:
                st.caption("⚠️ Error displaying fitted params")
                # Clear corrupted data
                st.session_state.fitted_params = {}
    
    # Load Session in Sidebar
    st.markdown("---")
    st.subheader("📂 Load Session")
    uploaded_session_sidebar = st.file_uploader("Upload session file", type=['json'], key='session_upload_sidebar')
    
    if uploaded_session_sidebar is not None:
        import json
        try:
            session_data = json.load(uploaded_session_sidebar)
            
            # Show preview
            st.caption(f"Session ID: {session_data.get('session_id', 'Unknown')}")
            st.caption(f"Saved: {session_data.get('timestamp', 'Unknown')[:10]}")
            
            if st.button("✅ Load Session", type="primary"):
                # Restore aquifer properties
                st.session_state.aquifer_properties = session_data['aquifer_properties']
                st.session_state.student_id = session_data['session_id']
                
                # Restore wells
                st.session_state.wells_Pumping = session_data.get('wells_Pumping', [])
                st.session_state.wells_Injection = session_data.get('wells_Injection', [])
                st.session_state.wells_Monitoring = session_data.get('wells_Monitoring', [])
                
                # Restore polygons
                st.session_state.polygons = session_data.get('polygons', [])
                
                # Restore fitted parameters if available
                if 'fitted_parameters' in session_data:
                    loaded_params = session_data['fitted_parameters']
                    
                    # Check if it's the old format (single dict) or new format (dict of dicts)
                    if loaded_params and isinstance(loaded_params, dict):
                        # Check if it's old format by looking for 'pumping_well' key directly
                        if 'pumping_well' in loaded_params:
                            # Old format - convert to new format
                            pump_well_label = loaded_params.get('pumping_well', 'P1')
                            st.session_state.fitted_params = {pump_well_label: loaded_params}
                        else:
                            # New format - already dict of dicts
                            st.session_state.fitted_params = loaded_params
                    else:
                        st.session_state.fitted_params = {}
                
                st.success(f"✅ Loaded!")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Quick save button
    st.markdown("---")
    st.subheader("💾 Save Session")
    
    if st.button("💾 Download Session"):
        import json
        import datetime
        
        session_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'session_id': st.session_state.student_id,
            'aquifer_properties': st.session_state.aquifer_properties,
            'wells_Pumping': st.session_state.wells_Pumping,
            'wells_Injection': st.session_state.wells_Injection,
            'wells_Monitoring': st.session_state.wells_Monitoring,
            'polygons': st.session_state.get('polygons', []),
        }
        
        # Add fitted parameters if they exist and are a dictionary
        if 'fitted_params' in st.session_state and isinstance(st.session_state.fitted_params, dict):
            session_data['fitted_parameters'] = st.session_state.fitted_params
        
        session_json = json.dumps(session_data, indent=2)
        
        st.download_button(
            "⬇️ Download",
            data=session_json,
            file_name=f"session_{st.session_state.student_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Header
st.title("🏞️ Well Placement and Pumping Test Simulator")
st.markdown("*An interactive tool for groundwater hydraulics education*")

st.markdown("---")

# Conversion factor
PIXEL_TO_METER = 10.0

# Initialize aquifer properties
if 'aquifer_properties' not in st.session_state:
    np.random.seed()
    st.session_state.aquifer_properties = {
        'T': np.random.uniform(200, 800),
        'S': 10.**np.random.uniform(-3, -2),
        'Sy': np.random.uniform(0.01, 0.25),
        'b': np.random.uniform(20, 100),
        'anis': np.random.uniform(0.01, 0.1),
        'rwell': np.random.uniform(0.1, 0.3)
    }
    st.session_state.student_id = np.random.randint(1000, 9999)

# Initialize well storage
for key in ["wells_Pumping", "wells_Injection", "wells_Monitoring"]:
    if key not in st.session_state:
        st.session_state[key] = []

# Initialize polygon storage
if 'polygons' not in st.session_state:
    st.session_state.polygons = []

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Add Wells & Areas", "🔬 Pumping Test", "⛏️ Dewatering", "🌍 Regional Impacts", "📚 Background and Instructions"])

# ============================================================================
# ============================================================================
# TAB 1: ADD WELLS & AREAS
# ============================================================================
with tab1:
    st.header("🗺️ Add Wells & Areas")
    
    # Define predefined areas of interest (in meters)
    if 'polygons' not in st.session_state or len(st.session_state.polygons) == 0:
        st.session_state.polygons = [
            {
                "name": "Mine Pit Max Extent",
                "points_meter": [
                    (856.2, 4488.2), (957.8, 4403.3), (1016.4, 4281.5), (1041.3, 4135.0),
                    (1041.4, 3976.2), (1025.9, 3817.7), (1003.1, 3669.7), (976.8, 3531.6),
                    (949.5, 3399.4), (923.5, 3268.7), (901.2, 3135.3), (884.2, 2996.9),
                    (871.1, 2855.9), (860.3, 2715.5), (850.2, 2579.0), (839.0, 2449.7),
                    (825.7, 2326.2), (810.7, 2200.7), (793.9, 2064.7), (775.6, 1910.0),
                    (755.7, 1729.8), (730.5, 1542.5), (693.1, 1386.3), (636.6, 1300.2),
                    (555.0, 1320.9), (463.5, 1435.8), (399.3, 1580.0), (374.8, 1718.1),
                    (370.9, 1852.5), (368.3, 1987.7), (362.5, 2125.0), (356.9, 2263.8),
                    (355.2, 2403.5), (361.0, 2543.5), (376.7, 2683.2), (400.2, 2821.8),
                    (427.8, 2958.8), (455.8, 3093.2), (480.3, 3224.4), (498.6, 3352.9),
                    (510.2, 3482.9), (514.9, 3619.0), (512.7, 3765.9), (503.7, 3927.9),
                    (493.3, 4100.3), (495.5, 4264.9), (524.2, 4402.8), (594.0, 4494.9),
                    (716.0, 4523.4), (856.2, 4488.2)
                ],
                "area_m2": 1406208,
                "color": "#FF6B6B"
            },
            {
                "name": "North Pit",
                "points_meter": [
                    (579.3, 3698.5), (610.3, 3650.0), (623.7, 3575.3), (627.9, 3488.3),
                    (631.6, 3402.8), (643.4, 3332.8), (671.6, 3292.1), (723.9, 3293.7),
                    (788.5, 3335.7), (836.5, 3402.8), (845.4, 3480.3), (831.2, 3559.8),
                    (822.6, 3634.7), (848.4, 3698.5), (937.4, 3744.5), (936.8, 3863.1),
                    (941.6, 3969.3), (956.4, 4082.2), (953.6, 4203.7), (916.3, 4312.3),
                    (837.3, 4390.0), (722.6, 4423.6), (622.4, 4399.0), (569.9, 4313.9),
                    (559.7, 4188.2), (580.8, 4051.9), (603.1, 3939.2), (591.2, 3827.9)
                ],
                "area_m2": 325939,
                "color": "#4169E1"
            },
            {
                "name": "South Pit",
                "points_meter": [
                    (439.7, 2061.5), (465.8, 1965.6), (464.2, 1888.6), (444.9, 1789.4),
                    (446.4, 1668.2), (488.0, 1532.7), (560.7, 1420.7), (629.4, 1399.9),
                    (674.8, 1484.6), (697.1, 1636.7), (693.4, 1808.0), (676.7, 1924.7),
                    (709.1, 2011.7), (736.7, 2126.6), (719.5, 2178.2), (711.4, 2251.6),
                    (706.9, 2335.5), (700.4, 2418.4), (686.4, 2489.0), (659.3, 2535.8),
                    (613.5, 2547.3), (550.8, 2517.1), (498.3, 2456.3), (484.5, 2379.7),
                    (493.2, 2300.5), (484.6, 2230.8), (418.5, 2182.9)
                ],
                "area_m2": 251257,
                "color": "#228B22"
            },
            {
                "name": "GDE (Riparian Zone)",
                "points_meter": [
                    (6200, 3800), (6500, 3750), (6700, 3650), (6850, 3500),
                    (6900, 3300), (6850, 3100), (6700, 2950), (6500, 2900),
                    (6300, 2950), (6150, 3100), (6100, 3300), (6150, 3500),
                    (6200, 3650)
                ],
                "area_m2": 298500,
                "color": "#00CED1"
            }
        ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Site Map with Grid")
        
        # Create figure with grid
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Set limits (in meters)
        x_max, y_max = 10000, 6000
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        
        # Add major grid lines every 1000 meters
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.4, color='gray')
        ax.set_xticks(np.arange(0, x_max+1, 1000))
        ax.set_yticks(np.arange(0, y_max+1, 1000))
        
        # Add minor grid lines every 500 meters
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.25, color='gray')
        ax.set_xticks(np.arange(0, x_max+1, 500), minor=True)
        ax.set_yticks(np.arange(0, y_max+1, 500), minor=True)
        
        # Plot predefined areas
        for area in st.session_state.polygons:
            polygon = MPLPolygon(area['points_meter'], alpha=0.2, facecolor=area.get('color', '#FF6B6B'), 
                                edgecolor=area.get('color', '#FF6B6B'), linewidth=2.5)
            ax.add_patch(polygon)
            # Add label at centroid
            xs = [p[0] for p in area['points_meter']]
            ys = [p[1] for p in area['points_meter']]
            centroid_x = sum(xs) / len(xs)
            centroid_y = sum(ys) / len(ys)
            ax.text(centroid_x, centroid_y, area['name'], 
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=area.get('color', '#FF6B6B'), linewidth=2))
        
        # Plot wells
        type_keys = {
            "Pumping": "wells_Pumping",
            "Injection": "wells_Injection",
            "Monitoring": "wells_Monitoring"
        }
        
        pumping_plotted = False
        injection_plotted = False
        monitoring_plotted = False
        
        for well_type, key in type_keys.items():
            if key in st.session_state and st.session_state[key]:
                for well in st.session_state[key]:
                    if well_type == 'Pumping':
                        ax.plot(well['x'], well['y'], 'ro', markersize=14, markeredgecolor='darkred', 
                               markeredgewidth=2.5, label='Pumping' if not pumping_plotted else '')
                        ax.text(well['x'], well['y']+200, well['label'], fontsize=10, ha='center', 
                               fontweight='bold', color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                        pumping_plotted = True
                    elif well_type == 'Injection':
                        ax.plot(well['x'], well['y'], 'go', markersize=14, markeredgecolor='darkgreen', 
                               markeredgewidth=2.5, label='Injection' if not injection_plotted else '')
                        ax.text(well['x'], well['y']+200, well['label'], fontsize=10, ha='center', 
                               fontweight='bold', color='darkgreen',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                        injection_plotted = True
                    else:  # Monitoring
                        ax.plot(well['x'], well['y'], 'bs', markersize=12, markeredgecolor='darkblue', 
                               markeredgewidth=2.5, label='Monitoring' if not monitoring_plotted else '')
                        ax.text(well['x'], well['y']+200, well['label'], fontsize=10, ha='center', 
                               fontweight='bold', color='darkblue',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                        monitoring_plotted = True
        
        ax.set_xlabel('X Coordinate (m)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Coordinate (m)', fontsize=13, fontweight='bold')
        ax.set_title('Site Map - Wells and Areas of Interest', fontsize=15, fontweight='bold', pad=15)
        ax.set_aspect('equal')
        
        # Create custom legend if any wells exist
        if pumping_plotted or injection_plotted or monitoring_plotted:
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("➕ Add New Well")
        
        with st.form("add_well_form"):
            well_x = st.number_input("X Coordinate (m)", min_value=0, max_value=10000, value=5000, step=100)
            well_y = st.number_input("Y Coordinate (m)", min_value=0, max_value=6000, value=3000, step=100)
            well_type = st.selectbox("Well Type", ["Pumping", "Injection", "Monitoring"])
            
            if st.form_submit_button("Add Well", use_container_width=True, type="primary"):
                # Generate label
                type_keys = {
                    "Pumping": "wells_Pumping",
                    "Injection": "wells_Injection",
                    "Monitoring": "wells_Monitoring"
                }
                type_prefix = {"Pumping": "P", "Injection": "I", "Monitoring": "M"}
                
                key = type_keys[well_type]
                if key not in st.session_state:
                    st.session_state[key] = []
                
                prefix = type_prefix[well_type]
                type_count = len(st.session_state[key])
                new_label = f"{prefix}{type_count + 1}"
                
                st.session_state[key].append({
                    "x": float(well_x),
                    "y": float(well_y),
                    "type": well_type,
                    "label": new_label
                })
                st.success(f"✅ Added {new_label}")
                st.rerun()
        
        st.markdown("---")
        st.subheader("📋 Current Wells")
        
        # Display wells by type
        type_keys = {
            "Pumping": "wells_Pumping",
            "Injection": "wells_Injection",
            "Monitoring": "wells_Monitoring"
        }
        
        total_wells = 0
        all_wells = []
        
        for well_type, key in type_keys.items():
            if key in st.session_state and st.session_state[key]:
                total_wells += len(st.session_state[key])
                all_wells.extend(st.session_state[key])
        
        if total_wells > 0:
            st.metric("Total Wells", total_wells)
            
            # Create editable dataframe
            wells_df = pd.DataFrame(all_wells)
            
            edited_df = st.data_editor(
                wells_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "x": st.column_config.NumberColumn("X (m)", min_value=0, max_value=10000, step=100, format="%d"),
                    "y": st.column_config.NumberColumn("Y (m)", min_value=0, max_value=6000, step=100, format="%d"),
                    "type": st.column_config.SelectboxColumn("Type", options=["Pumping", "Injection", "Monitoring"]),
                    "label": st.column_config.TextColumn("Label", width="small")
                },
                key="wells_editor"
            )
            
            # Update session state from edited dataframe
            if not edited_df.equals(wells_df):
                # Clear all well lists
                for key in type_keys.values():
                    st.session_state[key] = []
                
                # Repopulate from edited dataframe
                for _, row in edited_df.iterrows():
                    well_type = row['type']
                    key = type_keys[well_type]
                    st.session_state[key].append({
                        "x": float(row['x']),
                        "y": float(row['y']),
                        "type": well_type,
                        "label": row['label']
                    })
                st.rerun()
            
            # Delete well option
            st.markdown("##### 🗑️ Remove Well")
            all_labels = [w['label'] for w in all_wells]
            well_to_delete = st.selectbox(
                "Select well to remove:",
                options=all_labels,
                key="delete_well_selector"
            )
            
            if st.button("Remove Selected Well", use_container_width=True, type="secondary"):
                # Find and remove the well
                for well_type, key in type_keys.items():
                    if key in st.session_state:
                        st.session_state[key] = [w for w in st.session_state[key] if w['label'] != well_to_delete]
                st.success(f"✅ Removed {well_to_delete}")
                st.rerun()
        else:
            st.info("No wells placed yet. Add wells using the form above.")
        
        st.markdown("---")
        st.subheader("📐 Predefined Areas")
        for area in st.session_state.polygons:
            area_ha = area['area_m2'] / 10000  # Convert to hectares
            st.markdown(f"**{area['name']}**: {len(area['points_meter'])} vertices, {area_ha:.1f} ha")
        
        if st.button("🗑️ Clear All Wells", use_container_width=True):
            for key in type_keys.values():
                st.session_state[key] = []
            st.rerun()

# TAB 2: PUMPING TEST (UNCHANGED)
# ============================================================================
with tab2:
    st.header("Pumping Test Simulation")
    
    # Student ID
    st.success(f"🎓 **Session ID:** {st.session_state.student_id}")
    
    # Aquifer properties
    with st.expander("🔍 View True Aquifer Properties", expanded=False):
        st.warning("⚠️ **For instructors/verification only!** Students should estimate these from test data.")
        
        props = st.session_state.aquifer_properties
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transmissivity (T)", f"{props['T']:.1f} m²/day")
            st.metric("Storativity (S)", f"{props['S']:.2e}")
        with col2:
            st.metric("Specific Yield (Sy)", f"{props['Sy']:.3f}")
            st.metric("Thickness (b)", f"{props['b']:.1f} m")
        with col3:
            st.metric("Anisotropy (Kz/Kr)", f"{props['anis']:.3f}")
            st.metric("Well Radius", f"{props['rwell']:.3f} m")
        
        K = props['T'] / props['b']
        st.metric("**Hydraulic Conductivity (K)**", f"{K:.2f} m/day")

    # Get properties
    T = st.session_state.aquifer_properties['T']
    S = st.session_state.aquifer_properties['S']
    Sy = st.session_state.aquifer_properties['Sy']
    b = st.session_state.aquifer_properties['b']
    anis = st.session_state.aquifer_properties['anis']
    rwell = st.session_state.aquifer_properties['rwell']

    # Well selection
    pumping_wells = st.session_state["wells_Pumping"]
    monitoring_wells = st.session_state["wells_Monitoring"]

    col1, col2 = st.columns(2)
    
    with col1:
        if pumping_wells:
            selected_pumping = st.selectbox(
                "🔵 Pumping Well",
                options=[f"{w['label']} ({w['x']:.0f}, {w['y']:.0f}) m" for w in pumping_wells]
            )
            pumping_well_data = pumping_wells[[f"{w['label']} ({w['x']:.0f}, {w['y']:.0f}) m" 
                                               for w in pumping_wells].index(selected_pumping)]
        else:
            st.error("❌ No pumping wells available")
            pumping_well_data = None

    with col2:
        if monitoring_wells:
            selected_monitoring = st.multiselect(
                "🟠 Monitoring Wells",
                options=[f"{w['label']} ({w['x']:.0f}, {w['y']:.0f}) m" for w in monitoring_wells]
            )
            monitoring_well_data = [w for w in monitoring_wells 
                                   if f"{w['label']} ({w['x']:.0f}, {w['y']:.0f}) m" in selected_monitoring]
        else:
            st.error("❌ No monitoring wells available")
            monitoring_well_data = []

    # Test parameters
    st.subheader("⚙️ Test Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        Q = st.number_input("Pumping Rate (m³/day)", value=1000.0, min_value=1.0, step=100.0)
        ton = st.number_input("Start Time (days)", value=0.01, min_value=0.01)
    with col2:
        t_end = st.number_input("End Time (days)", value=10.0, min_value=0.01, step=1.0)
        n_times = st.number_input("Time Steps", value=50, min_value=10, max_value=200, step=10)
    with col3:
        aquifer_type = st.radio("Aquifer Type", ["Unconfined (Neuman)", "Confined (Theis)"])

    time_values = np.logspace(np.log10(ton), np.log10(t_end), n_times)

    # Run simulation
    if st.button("▶️ Run Simulation", type="primary"):
        if not pumping_well_data or not monitoring_well_data:
            st.error("⚠️ Select at least one pumping and one monitoring well!")
        else:
            with st.spinner('🔄 Computing drawdown...'):
                # Create well object
                if "Unconfined" in aquifer_type:
                    well = Neuman_well(
                        x=pumping_well_data['x'], y=pumping_well_data['y'],
                        Q=Q, ton=ton, T=T, S=S, b=b, anis=anis, Sy=Sy, rwell=rwell
                    )
                else:
                    well = Theis_well(
                        x=pumping_well_data['x'], y=pumping_well_data['y'],
                        Q=Q, ton=ton, T=T, S=S, rwell=rwell
                    )

                # Calculate drawdown with progress bar
                results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_calcs = len(monitoring_well_data) * len(time_values)
                calc_count = 0
                
                for mon_well in monitoring_well_data:
                    status_text.text(f"Computing drawdown at {mon_well['label']}...")
                    drawdowns = []
                    for t in time_values:
                        s = well.s(mon_well['x'], mon_well['y'], t)
                        drawdowns.append(s)
                        calc_count += 1
                        progress_bar.progress(calc_count / total_calcs)
                    results[mon_well['label']] = drawdowns
                
                progress_bar.empty()
                status_text.empty()
                
                # Store results in session state
                st.session_state['simulation_results'] = results
                st.session_state['simulation_times'] = time_values
                st.session_state['pumping_well_label'] = pumping_well_data['label']
                st.session_state['monitoring_well_data'] = monitoring_well_data

            st.success("✅ Simulation complete!")
    
    # Display results if available
    if 'simulation_results' in st.session_state:
        results = st.session_state['simulation_results']
        time_values = st.session_state['simulation_times']
        pumping_label = st.session_state['pumping_well_label']
        monitoring_well_data = st.session_state['monitoring_well_data']
        
        # Results
        st.subheader("📊 Results")
        
        # Create DataFrame
        df_results = pd.DataFrame(results, index=time_values)
        df_results.index.name = 'Time (days)'
        
        # Interactive plot controls - DEFINE THESE FIRST
        st.markdown("### 📈 Interactive Plot")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            x_scale = st.radio("X-axis scale", ["Linear", "Log"], key="plot_x", horizontal=True, index=1)
        with col2:
            y_scale = st.radio("Y-axis scale", ["Linear", "Log"], key="plot_y", horizontal=True, index=1)
        with col3:
            if st.button("🔄 Update Plot"):
                st.rerun()
        
        # Theis Curve Fitting Section
        st.markdown("### 🎯 Curve Fitting Exercise")
        st.info("💡 Adjust the sliders to fit a Theis curve to your data. Try to match the simulated points!")
        
        with st.expander("⚙️ Theis Curve Fitting Parameters", expanded=True):
            # Get initial values from saved fitted params for THIS pumping well if available
            pumping_well_key = pumping_well_data['label']
            if 'fitted_params' in st.session_state and pumping_well_key in st.session_state.fitted_params:
                saved_params = st.session_state.fitted_params[pumping_well_key]
                init_T = saved_params.get('T', 500.0)
                init_S = saved_params.get('S', 1e-2)
                init_Q = saved_params.get('Q', Q)
                st.info(f"ℹ️ Loaded previous fit for {pumping_well_key}")
            else:
                init_T = 500.0
                init_S = 1e-3
                init_Q = Q
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fit_T = st.slider(
                    "Transmissivity (T) [m²/day]",
                    min_value=100.0,
                    max_value=1000.0,
                    value=float(init_T),
                    step=10.0,
                    help="Adjust to match the slope of the curve"
                )
            
            with col2:
                fit_S = st.slider(
                    "Storativity (S)",
                    min_value=1e-3,
                    max_value=3e-1,
                    value=float(init_S),
                    step=1e-3,
                    format="%.2e",
                    help="Adjust to shift the curve horizontally"
                )
            
            with col3:
                fit_Q = st.slider(
                    "Pumping Rate (Q) [m³/day]",
                    min_value=100.0,
                    max_value=5000.0,
                    value=float(init_Q),
                    step=50.0,
                    help="Match the pumping rate used in simulation"
                )
            
            # Option to select which monitoring well to fit
            selected_fit_well = st.selectbox(
                "Select monitoring well to fit:",
                options=list(results.keys()),
                help="Choose which well's data to fit the Theis curve to"
            )
            
            # Get the monitoring well coordinates
            fit_well_data = [w for w in monitoring_well_data if w['label'] == selected_fit_well][0]
            
            # Calculate distance between pumping and monitoring well
            fit_distance = np.sqrt((fit_well_data['x'] - pumping_well_data['x'])**2 + 
                                  (fit_well_data['y'] - pumping_well_data['y'])**2)
            
            # Calculate Theis curve with fitted parameters
            fit_theis_well = Theis_well(
                x=pumping_well_data['x'],
                y=pumping_well_data['y'],
                Q=fit_Q,
                ton=ton,
                T=fit_T,
                S=fit_S,
                rwell=rwell
            )
            
            # Calculate fitted drawdowns
            fitted_drawdowns = []
            for t in time_values:
                s = fit_theis_well.s(fit_well_data['x'], fit_well_data['y'], t)
                fitted_drawdowns.append(s)
            
            # Calculate RMSE (Root Mean Square Error)
            observed = np.array(results[selected_fit_well])
            predicted = np.array(fitted_drawdowns)
            rmse = np.sqrt(np.mean((observed - predicted)**2))
            
            # Save fitted parameters to session state (keyed by pumping well)
            if 'fitted_params' not in st.session_state:
                st.session_state.fitted_params = {}
            
            pumping_well_key = pumping_well_data['label']
            st.session_state.fitted_params[pumping_well_key] = {
                'T': fit_T,
                'S': fit_S,
                'Q': fit_Q,
                'monitoring_well': selected_fit_well,
                'distance': fit_distance,
                'rmse': rmse,
                'x_scale': x_scale,
                'y_scale': y_scale,
                'pumping_well': pumping_well_data['label']
            }
            
            # Display fit quality
            st.metric("Root Mean Square Error (RMSE)", f"{rmse:.4f} m", 
                     help="Lower values indicate better fit. Good fit typically < 0.1 m")
        
        # Single plot with user-selected scales
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot simulated data as dots only
        for label in results.keys():
            alpha_val = 1.0 if label == selected_fit_well else 0.3
            ax.plot(time_values, results[label], 'o', markersize=6, 
                   label=f'{label} (Data)', alpha=alpha_val)
        
        # Get data limits before adding fitted curve
        all_data_values = [val for vals in results.values() for val in vals]
        y_min_data = min(all_data_values)
        y_max_data = max(all_data_values)
        y_margin = (y_max_data - y_min_data) * 0.1  # 10% margin
        
        x_min_data = min(time_values)
        x_max_data = max(time_values)
        x_margin = (x_max_data - x_min_data) * 0.05
        
        # Plot fitted Theis curve as a line
        ax.plot(time_values, fitted_drawdowns, '-', linewidth=2.5, 
               label=f'Fitted Theis (T={fit_T:.0f}, S={fit_S:.2e})', 
               color='red', alpha=0.8)
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Drawdown (m)', fontsize=12)
        
        # Calculate distance for title
        fit_distance = np.sqrt((fit_well_data['x'] - pumping_well_data['x'])**2 + 
                              (fit_well_data['y'] - pumping_well_data['y'])**2)
        
        ax.set_title(f'Drawdown vs Time (Pumping: {pumping_label}, Fitting: {selected_fit_well}, r={fit_distance:.1f}m)', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, which="both" if (x_scale == "Log" or y_scale == "Log") else "major", alpha=0.3)
        
        # Set scale based on user selection
        if x_scale == "Log":
            ax.set_xscale('log')
        if y_scale == "Log":
            ax.set_yscale('log')
        
        # IMPORTANT: Set axis limits AFTER setting scale to ensure they stick
        # Always set limits based on simulated data range
        ax.set_ylim([y_min_data - y_margin, y_max_data + y_margin])
        ax.set_xlim([x_min_data - x_margin, x_max_data + x_margin])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add scale description
        scale_info = f"**Current view:** {x_scale} X-axis, {y_scale} Y-axis"
        if x_scale == "Log" and y_scale == "Log":
            scale_info += " (Good for type curve matching)"
        elif x_scale == "Log" and y_scale == "Linear":
            scale_info += " (Semi-log plot for Cooper-Jacob method)"
        st.info(scale_info)
        
        # Parameter comparison section
        with st.expander("📊 Compare Your Fit to True Values", expanded=False):
            st.warning("⚠️ Expand this only after you're satisfied with your fit!")
            
            true_T = st.session_state.aquifer_properties['T']
            true_S = st.session_state.aquifer_properties['S']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Transmissivity (T)")
                st.metric("Your estimate", f"{fit_T:.1f} m²/day")
                st.metric("True value", f"{true_T:.1f} m²/day")
                T_error = abs(fit_T - true_T) / true_T * 100
                st.metric("Relative error", f"{T_error:.1f}%")
            
            with col2:
                st.markdown("#### Storativity (S)")
                st.metric("Your estimate", f"{fit_S:.2e}")
                st.metric("True value", f"{true_S:.2e}")
                S_error = abs(fit_S - true_S) / true_S * 100
                st.metric("Relative error", f"{S_error:.1f}%")
            
            # Grading feedback
            if T_error < 10 and S_error < 20:
                st.success("🎉 Excellent fit! Your parameters are very close to the true values.")
            elif T_error < 20 and S_error < 50:
                st.info("👍 Good fit! Your parameters are reasonable.")
            else:
                st.warning("🤔 Keep adjusting! Try to get closer to the true values.")
        
        # Download fitted parameters
        st.markdown("### 💾 Save Your Work")
        col1, col2 = st.columns(2)
        
        with col1:
            # Create fitted parameters report for ALL pumping wells
            fit_reports = []
            
            if 'fitted_params' in st.session_state:
                for pump_well, params in st.session_state.fitted_params.items():
                    true_T = st.session_state.aquifer_properties['T']
                    true_S = st.session_state.aquifer_properties['S']
                    
                    fit_reports.append({
                        'Session_ID': st.session_state.student_id,
                        'Pumping_Well': pump_well,
                        'Monitoring_Well': params.get('monitoring_well', 'N/A'),
                        'Distance_m': params.get('distance', 0),
                        'Fitted_T_m2day': params.get('T', 0),
                        'Fitted_S': params.get('S', 0),
                        'Fitted_Q_m3day': params.get('Q', 0),
                        'RMSE_m': params.get('rmse', 0),
                        'True_T_m2day': true_T,
                        'True_S': true_S,
                        'T_Error_percent': abs(params.get('T', 0) - true_T) / true_T * 100,
                        'S_Error_percent': abs(params.get('S', 0) - true_S) / true_S * 100,
                        'X_Scale': params.get('x_scale', 'N/A'),
                        'Y_Scale': params.get('y_scale', 'N/A')
                    })
            
            # Add current fit if not already in the list
            current_pump_key = pumping_label
            if current_pump_key not in [r['Pumping_Well'] for r in fit_reports]:
                fit_reports.append({
                    'Session_ID': st.session_state.student_id,
                    'Pumping_Well': pumping_label,
                    'Monitoring_Well': selected_fit_well,
                    'Distance_m': fit_distance,
                    'Fitted_T_m2day': fit_T,
                    'Fitted_S': fit_S,
                    'Fitted_Q_m3day': fit_Q,
                    'RMSE_m': rmse,
                    'True_T_m2day': st.session_state.aquifer_properties['T'],
                    'True_S': st.session_state.aquifer_properties['S'],
                    'T_Error_percent': abs(fit_T - st.session_state.aquifer_properties['T']) / st.session_state.aquifer_properties['T'] * 100,
                    'S_Error_percent': abs(fit_S - st.session_state.aquifer_properties['S']) / st.session_state.aquifer_properties['S'] * 100,
                    'X_Scale': x_scale,
                    'Y_Scale': y_scale
                })
            
            # Convert to DataFrame for nice formatting
            if fit_reports:
                fit_df = pd.DataFrame(fit_reports)
                fit_csv = fit_df.to_csv(index=False)
                
                st.download_button(
                    "📥 Download All Fitted Parameters (CSV)",
                    data=fit_csv,
                    file_name=f"fitted_parameters_{st.session_state.student_id}.csv",
                    mime="text/csv",
                    help=f"Download fitted parameters for all {len(fit_reports)} pumping well(s)"
                )
            else:
                st.info("No fitted parameters to download yet")
        
        with col2:
            # Save entire session state
            import json
            import datetime
            
            session_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'session_id': st.session_state.student_id,
                'aquifer_properties': st.session_state.aquifer_properties,
                'wells_Pumping': st.session_state.wells_Pumping,
                'wells_Injection': st.session_state.wells_Injection,
                'wells_Monitoring': st.session_state.wells_Monitoring,
                'fitted_parameters': {
                    'T': fit_T,
                    'S': fit_S,
                    'Q': fit_Q,
                    'pumping_well': pumping_label,
                    'monitoring_well': selected_fit_well,
                    'x_scale': x_scale,
                    'y_scale': y_scale
                }
            }
            
            session_json = json.dumps(session_data, indent=2)
            
            st.download_button(
                "💾 Download Complete Session (JSON)",
                data=session_json,
                file_name=f"session_{st.session_state.student_id}.json",
                mime="application/json",
                help="Save your entire session including wells and parameters"
            )
        
        # Data table
        with st.expander("📋 View Data Table", expanded=False):
            st.dataframe(df_results.style.format("{:.4f}"), use_container_width=True)
        
        # Summary
        st.subheader("📈 Summary Statistics")
        summary_data = []
        for mon_well in monitoring_well_data:
            distance = np.sqrt((mon_well['x'] - pumping_well_data['x'])**2 + 
                             (mon_well['y'] - pumping_well_data['y'])**2)
            max_dd = max(results[mon_well['label']])
            summary_data.append({
                'Well': mon_well['label'],
                'Distance (m)': f"{distance:.1f}",
                'Max Drawdown (m)': f"{max_dd:.4f}",
                'Time to Max (days)': f"{time_values[np.argmax(results[mon_well['label']])]:.2f}"
            })
        
        st.table(pd.DataFrame(summary_data))
        
        # Download
        csv = df_results.to_csv()
        st.download_button(
            "💾 Download CSV",
            data=csv,
            file_name=f"pumping_test_{st.session_state.student_id}.csv",
            mime="text/csv"
        )

# ============================================================================
# TAB 3: AREA ANALYSIS - WITH TIME-SERIES PLOTS
# ============================================================================
with tab3:
    st.header("⛏️ Mine Dewatering Analysis")
    
    if not st.session_state.polygons:
        st.warning("⚠️ No areas defined yet. Go to 'Add Wells & Areas' tab to draw polygons.")
    else:
        st.info("💡 Predict water table elevation in mine pits over time. Initial water table: {INITIAL_WATER_TABLE}m")
        
        # Shared aquifer parameters (used in both Dewatering and Regional Impacts tabs)
        st.subheader("🌊 Shared Aquifer Parameters")
        st.info("💡 These parameters are shared between Dewatering and Regional Impacts tabs")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            shared_T = st.number_input("Transmissivity (T) [m²/day]", 
                                     value=500.0, min_value=1.0, max_value=10000.0, step=10.0,
                                     key="shared_T",
                                     help="Used in both Dewatering and Regional Impacts tabs")
        with col2:
            shared_S = st.number_input("Storativity (S)", 
                                     value=0.01, min_value=1e-3, max_value=0.5, 
                                     format="%.6f", step=1e-3,
                                     key="shared_S",
                                     help="Used in both Dewatering and Regional Impacts tabs")
        with col3:
            shared_rwell = st.number_input("Well Radius [m]", 
                                         value=0.15, min_value=0.01, max_value=1.0, step=0.01,
                                         key="shared_rwell",
                                         help="Used in both Dewatering and Regional Impacts tabs")
        
        # Note: Values are automatically stored in session state via the 'key' parameter above
        
        # Time-series parameters
        st.subheader("⏱️ Time-Series Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            area_end_time = st.number_input("End Time [years]", 
                                            value=20.0, min_value=0.1, max_value=100.0, step=1.0,
                                            help="Duration of analysis period (max 100 years)")
        with col2:
            area_n_times = st.number_input("Number of Time Steps", 
                                           value=100, min_value=20, max_value=500, step=10,
                                           help="More points = smoother curves but slower")
        with col3:
            grid_density = st.slider("Grid Density", min_value=10, max_value=50, value=20, step=5,
                                     help="Number of grid points per area dimension")
        with col4:
            pit_delay = st.number_input("Pit Start Delay [years]",
                                       value=0.0, min_value=0.0, max_value=20.0, step=0.5,
                                       help="Delay before mining starts (shifts bench progression)")
        
        # Convert years to days for calculations
        area_end_time_days = area_end_time * 365.25
        pit_delay_days = pit_delay * 365.25
        
        # Use shared parameters
        area_T = shared_T
        area_S = shared_S  
        area_rwell = shared_rwell
        
        # Well configuration
        st.subheader("🚰 Active Wells")
        
        use_placed_wells_area = st.checkbox("Use wells from 'Add Wells & Areas' tab", value=True, key="use_placed_area")
        
        if use_placed_wells_area:
            pumping_wells_list = st.session_state["wells_Pumping"]
            injection_wells_list = st.session_state["wells_Injection"]
            
            if not pumping_wells_list and not injection_wells_list:
                st.warning("⚠️ No wells placed. Add wells in the first tab.")
                active_wells = []
            else:
                st.success(f"Found {len(pumping_wells_list)} pumping and {len(injection_wells_list)} injection wells")
                
                active_wells = []
                
                # Pumping wells with recovery option
                if pumping_wells_list:
                    st.markdown("**Pumping Wells:**")
                    
                    for idx, well in enumerate(pumping_wells_list):
                        with st.expander(f"**{well['label']}** - ({well['x']:.0f}, {well['y']:.0f}) m", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                Q = st.number_input(f"Rate [m³/day]", 
                                                   value=4320.0, min_value=0.0, step=100.0,
                                                   key=f"area_Q_{well['label']}")
                            with col2:
                                ton = st.number_input(f"Start [days]", 
                                                     value=0.01, min_value=0.001, step=0.1,
                                                     key=f"area_ton_{well['label']}")
                            with col3:
                                include_recovery = st.checkbox(f"Include Recovery", value=False, 
                                                              key=f"area_recovery_{well['label']}",
                                                              help="Simulate when this well stops pumping")
                            with col4:
                                active = st.checkbox(f"Active", value=True, key=f"area_active_{well['label']}")
                            
                            if include_recovery:
                                toff = st.number_input(f"Stop Time [days]", 
                                                      value=max(100.0, ton+1.0), 
                                                      min_value=ton+0.01, step=1.0,
                                                      key=f"area_toff_{well['label']}",
                                                      help="When this well stops pumping")
                            else:
                                toff = None
                            
                            if active and Q > 0:
                                active_wells.append({
                                    'x': well['x'],
                                    'y': well['y'],
                                    'Q': Q,
                                    'ton': ton,
                                    'toff': toff,
                                    'label': well['label'],
                                    'type': 'Pumping'
                                })
                
                # Injection wells with stop option
                if injection_wells_list:
                    st.markdown("**Injection Wells:**")
                    
                    for idx, well in enumerate(injection_wells_list):
                        with st.expander(f"**{well['label']}** - ({well['x']:.0f}, {well['y']:.0f}) m", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                Q = st.number_input(f"Rate [m³/day]", 
                                                   value=500.0, min_value=0.0, step=100.0,
                                                   key=f"area_Q_{well['label']}")
                            with col2:
                                ton = st.number_input(f"Start [days]", 
                                                     value=0.01, min_value=0.001, step=0.1,
                                                     key=f"area_ton_{well['label']}")
                            with col3:
                                include_stop = st.checkbox(f"Include Stop", value=False, 
                                                          key=f"area_stop_{well['label']}",
                                                          help="Simulate when this well stops injecting")
                            with col4:
                                active = st.checkbox(f"Active", value=True, key=f"area_active_{well['label']}")
                            
                            if include_stop:
                                toff = st.number_input(f"Stop Time [days]", 
                                                      value=max(100.0, ton+1.0), 
                                                      min_value=ton+0.01, step=1.0,
                                                      key=f"area_toff_{well['label']}")
                            else:
                                toff = None
                            
                            if active and Q > 0:
                                active_wells.append({
                                    'x': well['x'],
                                    'y': well['y'],
                                    'Q': -Q,  # Negative for injection
                                    'ton': ton,
                                    'toff': toff,
                                    'label': well['label'],
                                    'type': 'Injection'
                                })
        else:
            st.info("Manual well entry not implemented for this tab. Use placed wells.")
            active_wells = []
        
        # Show active wells summary
        if active_wells:
            st.markdown("---")
            st.markdown("### 📋 Active Wells Summary")
            summary_wells = []
            for w in active_wells:
                recovery_status = "Yes" if w.get('toff') is not None else "No"
                toff_display = f"{w['toff']:.2f}" if w.get('toff') is not None else "N/A"
                summary_wells.append({
                    'Well': w['label'],
                    'Type': w['type'],
                    'Rate (m³/day)': f"{abs(w['Q']):.1f}",
                    'Start (days)': f"{w['ton']:.2f}",
                    'Stop (days)': toff_display,
                    'Recovery': recovery_status
                })
            st.table(pd.DataFrame(summary_wells))
        
        # Run analysis
        st.markdown("---")
        if st.button("🔬 Analyze Areas Over Time", type="primary"):
            if not active_wells:
                st.error("⚠️ No active wells configured!")
            else:
                with st.spinner('🔄 Computing drawdown time-series in areas...'):
                    # Create well objects with toff support
                    well_objects = []
                    for aw in active_wells:
                        well_obj = Theis_well(
                            x=aw['x'], y=aw['y'], Q=aw['Q'], ton=aw['ton'],
                            T=area_T, S=area_S, rwell=area_rwell,
                            toff=aw.get('toff', None)
                        )
                        well_objects.append(well_obj)
                    
                    # Create time array (linear spacing in days)
                    time_array = np.linspace(0.01, area_end_time_days, area_n_times)
                    
                    # Analyze each polygon over time
                    area_results = []
                    
                    progress_bar = st.progress(0)
                    total_steps = len(st.session_state.polygons) * len(time_array)
                    current_step = 0
                    
                    for poly_idx, poly in enumerate(st.session_state.polygons):
                        status_text = st.empty()
                        status_text.text(f"Analyzing {poly['name']} over time...")
                        
                        # Create grid within polygon bounding box
                        points = poly['points_meter']
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        x_grid = np.linspace(x_min, x_max, grid_density)
                        y_grid = np.linspace(y_min, y_max, grid_density)
                        
                        # Get grid points within polygon (once)
                        grid_points_in_poly = []
                        for x in x_grid:
                            for y in y_grid:
                                if point_in_polygon(x, y, points):
                                    grid_points_in_poly.append((x, y))
                        
                        if len(grid_points_in_poly) > 0:
                            # Time-series arrays for this area
                            min_drawdown_over_time = []
                            mean_drawdown_over_time = []
                            max_drawdown_over_time = []
                            
                            # Calculate drawdown at each time step
                            for t_idx, t in enumerate(time_array):
                                drawdown_at_t = []
                                
                                # Calculate drawdown at all grid points for this time
                                for (x, y) in grid_points_in_poly:
                                    total_dd = 0.0
                                    for well_obj in well_objects:
                                        total_dd += well_obj.s(x, y, t)
                                    drawdown_at_t.append(total_dd)
                                
                                # Store statistics
                                min_drawdown_over_time.append(np.min(drawdown_at_t))
                                mean_drawdown_over_time.append(np.mean(drawdown_at_t))
                                max_drawdown_over_time.append(np.max(drawdown_at_t))
                                
                                current_step += 1
                                if current_step % 10 == 0:
                                    progress_bar.progress(current_step / total_steps)
                            
                            area_results.append({
                                'name': poly['name'],
                                'area_m2': poly['area_m2'],
                                'n_points': len(grid_points_in_poly),
                                'time': time_array,
                                'min_drawdown': np.array(min_drawdown_over_time),
                                'mean_drawdown': np.array(mean_drawdown_over_time),
                                'max_drawdown': np.array(max_drawdown_over_time),
                                'grid_points': grid_points_in_poly
                            })
                        
                        status_text.empty()
                    
                    progress_bar.empty()
                    
                    # Store results
                    st.session_state['area_timeseries_results'] = area_results
                    st.session_state['area_analysis_wells'] = active_wells
                    st.session_state['area_end_time'] = area_end_time
                    
                st.success("✅ Time-series analysis complete!")
        
        # Display results
        if 'area_timeseries_results' in st.session_state:
            area_results = st.session_state['area_timeseries_results']
            analysis_wells = st.session_state.get('area_analysis_wells', [])
            end_time = st.session_state.get('area_end_time', 365.0)
            
            st.markdown("---")
            st.subheader(f"📊 Drawdown Time-Series Results (0 to {end_time:.1f} days)")
            
            # Summary statistics at final time
            st.markdown("### 📋 Final State Summary")
            summary_df = pd.DataFrame([{
                'Area': r['name'],
                'Size (m²)': f"{r['area_m2']:.0f}",
                'Sample Points': r['n_points'],
                'Final Min (m)': f"{r['min_drawdown'][-1]:.4f}",
                'Final Mean (m)': f"{r['mean_drawdown'][-1]:.4f}",
                'Final Max (m)': f"{r['max_drawdown'][-1]:.4f}",
                'Peak Mean (m)': f"{np.max(r['mean_drawdown']):.4f}",
                'Peak Time (days)': f"{r['time'][np.argmax(r['mean_drawdown'])]:.2f}"
            } for r in area_results])
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Time-series plots
            st.markdown("---")
            st.subheader("📈 Water Table Elevation Over Time")
            
            # Plot options
            col1, col2 = st.columns(2)
            with col1:
                plot_scale = st.radio("Time axis scale:", ["Linear", "Log"], horizontal=True, index=0)  # Default to Linear
            with col2:
                show_recovery_lines = st.checkbox("Show recovery events", value=True)
            
            # Get recovery events
            recovery_events = [w for w in analysis_wells if w.get('toff') is not None]
            
            # Create individual plots for North and South pits only
            for area_data in area_results:
                area_name = area_data['name']
                
                # Skip the main pit extent - only plot North and South pits
                if 'Max Extent' in area_name or 'Main Pit' in area_name:
                    continue
                
                # Only plot if it's North or South Pit
                if not (('North' in area_name and 'Pit' in area_name) or ('South' in area_name and 'Pit' in area_name)):
                    continue
                
                fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                
                time_days = area_data['time']
                time_years = time_days / 365.25  # Convert to years
                
                # Convert drawdown to water table elevation
                wt_min = INITIAL_WATER_TABLE - area_data['max_drawdown']  # max drawdown = min water table
                wt_mean = INITIAL_WATER_TABLE - area_data['mean_drawdown']
                wt_max = INITIAL_WATER_TABLE - area_data['min_drawdown']  # min drawdown = max water table
                
                # Plot water table elevation
                ax.fill_between(time_years, wt_min, wt_max, 
                               alpha=0.2, color='steelblue', label='Min-Max Range')
                ax.plot(time_years, wt_mean, '-', linewidth=3, color='darkblue', 
                       label='Mean Water Table', zorder=5)
                ax.plot(time_years, wt_min, '--', linewidth=2, color='steelblue', 
                       alpha=0.7, label='Minimum', zorder=4)
                ax.plot(time_years, wt_max, '--', linewidth=2, color='navy', 
                       alpha=0.7, label='Maximum', zorder=4)
                
                # Add bench progression overlay
                bench_data = None
                if 'North' in area_name and 'Pit' in area_name:
                    bench_data = NORTH_PIT_BENCHES
                    bench_label = 'North Pit Bench Progression'
                elif 'South' in area_name and 'Pit' in area_name:
                    bench_data = SOUTH_PIT_BENCHES
                    bench_label = 'South Pit Bench Progression'
                
                if bench_data is not None:
                    ax.plot(bench_data[:, 0] + pit_delay, bench_data[:, 1], 
                           'o-', linewidth=3.5, markersize=10, color='red', 
                           label=bench_label, zorder=10, markeredgecolor='darkred', markeredgewidth=2)
                
                # Add recovery event lines
                if show_recovery_lines and recovery_events:
                    for well in recovery_events:
                        if well.get('toff') is not None:
                            toff_years = well['toff'] / 365.25
                            ax.axvline(x=toff_years, color='orange', linestyle='--', 
                                     linewidth=2, alpha=0.6)
                            ax.text(toff_years, ax.get_ylim()[1]*0.95, 
                                   f"{well['label']}\nstops", 
                                   rotation=90, va='top', ha='right', fontsize=9, color='orange',
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Time (years)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Water Table Elevation (m)', fontsize=13, fontweight='bold')
                ax.set_title(f"{area_data['name']}: Water Table vs Bench Progression\n({area_data['area_m2']:.0f} m², {area_data['n_points']} sample points)", 
                           fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=11, framealpha=0.95)
                ax.grid(True, alpha=0.3, which='both' if plot_scale == 'Log' else 'major')
                
                if plot_scale == 'Log':
                    ax.set_xscale('log')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add interpretation help
                if bench_data is not None:
                    st.info(f"💡 **Success criterion**: Water table (blue) must stay BELOW bench elevation (red) at all times!")
            
            # Comparison plot - all areas on one plot (optional, skip Main Pit)
            st.markdown("---")
            st.subheader("📊 North vs South Pit Comparison")
            
            # Filter out main pit
            pit_results = [r for r in area_results if not ('Max Extent' in r['name'] or 'Main Pit' in r['name'])]
            
            if len(pit_results) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(pit_results)))
                
                for idx, area_data in enumerate(pit_results):
                    time_days = area_data['time']
                    time_years = time_days / 365.25
                    wt_mean = INITIAL_WATER_TABLE - area_data['mean_drawdown']
                    wt_min = INITIAL_WATER_TABLE - area_data['max_drawdown']
                    wt_max = INITIAL_WATER_TABLE - area_data['min_drawdown']
                    
                    ax.plot(time_years, wt_mean, '-', linewidth=3, 
                           color=colors[idx], label=f"{area_data['name']} (Mean WT)", alpha=0.8, zorder=5)
                    ax.fill_between(time_years, wt_min, wt_max, 
                                   alpha=0.15, color=colors[idx])
                
                # Add bench progression for both pits
                ax.plot(NORTH_PIT_BENCHES[:, 0] + pit_delay, NORTH_PIT_BENCHES[:, 1], 
                       'o-', linewidth=3, markersize=9, color='red', 
                       label='North Pit Benches', zorder=10, markeredgecolor='darkred', markeredgewidth=1.5)
                ax.plot(SOUTH_PIT_BENCHES[:, 0] + pit_delay, SOUTH_PIT_BENCHES[:, 1], 
                       's-', linewidth=3, markersize=9, color='orangered', 
                       label='South Pit Benches', zorder=10, markeredgecolor='darkred', markeredgewidth=1.5)
                
                # Add recovery lines
                if show_recovery_lines and recovery_events:
                    for well in recovery_events:
                        if well.get('toff') is not None:
                            toff_years = well['toff'] / 365.25
                            ax.axvline(x=toff_years, color='orange', linestyle='--', 
                                     linewidth=2, alpha=0.5)
                
                ax.set_xlabel('Time (years)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Mean Water Table Elevation (m)', fontsize=13, fontweight='bold')
                ax.set_title('Comparison: Water Table vs Bench Progression (Both Pits)', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=11, framealpha=0.95)
                ax.grid(True, alpha=0.3, which='both' if plot_scale == 'Log' else 'major')
                
                if plot_scale == 'Log':
                    ax.set_xscale('log')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("💡 **Visual check**: Water table lines (solid) should stay below bench progression markers at all times!")
            
            # Download results
            # Download results
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            # Create comprehensive CSV
            csv_data = []
            for area_data in area_results:
                for i, t in enumerate(area_data['time']):
                    csv_data.append({
                        'Area': area_data['name'],
                        'Area_m2': area_data['area_m2'],
                        'Time_days': t,
                        'Time_years': t / 365.25,
                        'Min_Drawdown_m': area_data['min_drawdown'][i],
                        'Mean_Drawdown_m': area_data['mean_drawdown'][i],
                        'Max_Drawdown_m': area_data['max_drawdown'][i],
                        'Min_WaterTable_m': INITIAL_WATER_TABLE - area_data['max_drawdown'][i],
                        'Mean_WaterTable_m': INITIAL_WATER_TABLE - area_data['mean_drawdown'][i],
                        'Max_WaterTable_m': INITIAL_WATER_TABLE - area_data['min_drawdown'][i]
                    })
            
            results_df = pd.DataFrame(csv_data)
            results_csv = results_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download Time-Series Data (CSV)",
                data=results_csv,
                file_name=f"area_timeseries_{st.session_state.student_id}.csv",
                mime="text/csv"
            )
            
            # Also offer summary table download
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download Summary Table (CSV)",
                data=summary_csv,
                file_name=f"area_summary_{st.session_state.student_id}.csv",
                mime="text/csv"
            )

# ============================================================================
# TAB 4: MINE DEWATERING - COMPLETE WITH RECOVERY SUPPORT
# ============================================================================
with tab4:
    st.header("🌍 Regional Impacts Analysis")
    
    st.info("💡 Visualize drawdown maps and monitor specific locations over time from multiple active wells (with optional recovery phases)")
    
    # Aquifer parameters
    st.subheader("🌊 Aquifer Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mine_T = st.number_input("Transmissivity (T) [m²/day]", 
                                 value=500.0, min_value=1.0, max_value=10000.0, step=10.0,
                                 key="mine_T")
    with col2:
        mine_S = st.number_input("Storativity (S)", 
                                 value=0.01, min_value=1e-3, max_value=0.5, 
                                 format="%.6f", step=1e-3,
                                 key="mine_S")
    with col3:
        mine_rwell = st.number_input("Well Radius [m]", 
                                     value=0.15, min_value=0.01, max_value=1.0, step=0.01,
                                     key="mine_rwell")
    with col4:
        aquifer_type_mine = st.selectbox("Aquifer Type", 
                                         ["Confined (Theis)", "Unconfined (Neuman)"],
                                         key="mine_aquifer_type")
    
    # Additional parameters for Neuman if selected
    if "Unconfined" in aquifer_type_mine:
        col1, col2, col3 = st.columns(3)
        with col1:
            mine_Sy = st.number_input("Specific Yield (Sy)", 
                                      value=0.15, min_value=0.01, max_value=0.5, step=0.01,
                                      key="mine_Sy")
        with col2:
            mine_b = st.number_input("Aquifer Thickness [m]", 
                                     value=50.0, min_value=1.0, max_value=500.0, step=5.0,
                                     key="mine_b")
        with col3:
            mine_anis = st.number_input("Anisotropy (Kz/Kr)", 
                                        value=0.1, min_value=0.001, max_value=1.0, step=0.01,
                                        key="mine_anis")
    
    # Well configuration
    st.subheader("🚰 Active Wells Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        use_placed_wells = st.checkbox("Use wells from 'Add Wells & Areas' tab", value=True, key="use_placed_mine")
    with col2:
        sync_dewatering = st.checkbox("📋 Sync configuration from Dewatering tab", value=True, key="sync_dewatering",
                                      help="Use same pump rates and timing as configured in Dewatering tab")
    
    active_wells_mine = []
    
    if use_placed_wells:
        pumping_wells_list = st.session_state["wells_Pumping"]
        injection_wells_list = st.session_state["wells_Injection"]
        
        if not pumping_wells_list and not injection_wells_list:
            st.warning("⚠️ No wells placed. Add wells in the first tab or uncheck the box above.")
        else:
            if sync_dewatering:
                st.info("📋 Using well configuration from Dewatering tab. Modify settings there to update both tabs.")
            else:
                st.success(f"Found {len(pumping_wells_list)} pumping and {len(injection_wells_list)} injection wells")
            
            # Pumping wells with recovery option
            if pumping_wells_list:
                st.markdown("**⚙️ Pumping Wells:**")
                
                # Create expandable sections for each well
                for idx, well in enumerate(pumping_wells_list):
                    # Check if we should sync from dewatering tab
                    if sync_dewatering:
                        # Get values from dewatering tab (area_* keys)
                        dewater_Q_key = f"area_Q_{well['label']}"
                        dewater_ton_key = f"area_ton_{well['label']}"
                        dewater_recovery_key = f"area_recovery_{well['label']}"
                        dewater_toff_key = f"area_toff_{well['label']}"
                        dewater_active_key = f"area_active_{well['label']}"
                        
                        # Use dewatering tab values if they exist, otherwise use defaults
                        Q = st.session_state.get(dewater_Q_key, 1000.0)
                        ton = st.session_state.get(dewater_ton_key, 0.01)
                        include_recovery = st.session_state.get(dewater_recovery_key, False)
                        active = st.session_state.get(dewater_active_key, True)
                        toff = st.session_state.get(dewater_toff_key, None) if include_recovery else None
                        
                        # Display as read-only info
                        with st.expander(f"**{well['label']}** - ({well['x']:.0f}, {well['y']:.0f}) m", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rate", f"{Q:.0f} m³/day")
                            with col2:
                                st.metric("Start", f"{ton:.2f} days")
                            with col3:
                                if include_recovery and toff is not None:
                                    st.metric("Stop", f"{toff:.0f} days")
                                else:
                                    st.metric("Stop", "Continuous")
                            if active:
                                st.success("✓ Active")
                            else:
                                st.warning("✗ Inactive")
                    else:
                        # Original configuration (user can modify)
                        with st.expander(f"**{well['label']}** - ({well['x']:.0f}, {well['y']:.0f}) m", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                Q = st.number_input(f"Rate [m³/day]", 
                                                   value=4320.0, min_value=0.0, step=100.0,
                                                   key=f"mine_Q_{well['label']}")
                            with col2:
                                ton = st.number_input(f"Start Time [days]", 
                                                     value=0.01, min_value=0.001, step=0.1,
                                                     key=f"mine_ton_{well['label']}")
                            with col3:
                                include_recovery = st.checkbox(f"Include Recovery", value=False, 
                                                              key=f"mine_recovery_{well['label']}",
                                                              help="Simulate when this well stops pumping")
                            with col4:
                                active = st.checkbox(f"Active", value=True, key=f"mine_active_{well['label']}")
                            
                            # Recovery time input
                            if include_recovery:
                                toff = st.number_input(f"Stop Time [days]", 
                                                      value=max(100.0, ton+1.0), 
                                                      min_value=ton+0.01, step=1.0,
                                                      key=f"mine_toff_{well['label']}",
                                                      help="When this well stops pumping (recovery begins)")
                            else:
                                toff = None
                    
                    if active and Q > 0:
                        active_wells_mine.append({
                            'x': well['x'],
                            'y': well['y'],
                            'Q': Q,
                            'ton': ton,
                            'toff': toff,
                            'label': well['label'],
                            'type': 'Pumping'
                        })
            
            # Injection wells
            if injection_wells_list:
                st.markdown("**⚙️ Injection Wells:**")
                
                for idx, well in enumerate(injection_wells_list):
                    # Check if we should sync from dewatering tab
                    if sync_dewatering:
                        # Get values from dewatering tab (area_* keys)
                        dewater_Q_key = f"area_Q_{well['label']}"
                        dewater_ton_key = f"area_ton_{well['label']}"
                        dewater_stop_key = f"area_stop_{well['label']}"
                        dewater_toff_key = f"area_toff_{well['label']}"
                        dewater_active_key = f"area_active_{well['label']}"
                        
                        # Use dewatering tab values if they exist, otherwise use defaults
                        Q = st.session_state.get(dewater_Q_key, 500.0)
                        ton = st.session_state.get(dewater_ton_key, 0.01)
                        include_stop = st.session_state.get(dewater_stop_key, False)
                        active = st.session_state.get(dewater_active_key, True)
                        toff = st.session_state.get(dewater_toff_key, None) if include_stop else None
                        
                        # Display as read-only info
                        with st.expander(f"**{well['label']}** - ({well['x']:.0f}, {well['y']:.0f}) m", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rate", f"{Q:.0f} m³/day")
                            with col2:
                                st.metric("Start", f"{ton:.2f} days")
                            with col3:
                                if include_stop and toff is not None:
                                    st.metric("Stop", f"{toff:.0f} days")
                                else:
                                    st.metric("Stop", "Continuous")
                            if active:
                                st.success("✓ Active")
                            else:
                                st.warning("✗ Inactive")
                    else:
                        # Original configuration (user can modify)
                        with st.expander(f"**{well['label']}** - ({well['x']:.0f}, {well['y']:.0f}) m", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                Q = st.number_input(f"Rate [m³/day]", 
                                                   value=500.0, min_value=0.0, step=100.0,
                                                   key=f"mine_Q_{well['label']}")
                            with col2:
                                ton = st.number_input(f"Start Time [days]", 
                                                     value=0.01, min_value=0.001, step=0.1,
                                                     key=f"mine_ton_{well['label']}")
                            with col3:
                                include_stop = st.checkbox(f"Include Stop", value=False, 
                                                              key=f"mine_recovery_{well['label']}",
                                                              help="Simulate when this well stops injecting")
                            with col4:
                                active = st.checkbox(f"Active", value=True, key=f"mine_active_{well['label']}")
                            
                            if include_stop:
                                toff = st.number_input(f"Stop Time [days]", 
                                                      value=max(100.0, ton+1.0), 
                                                      min_value=ton+0.01, step=1.0,
                                                      key=f"mine_toff_{well['label']}")
                            else:
                                toff = None
                    
                    if active and Q > 0:
                        active_wells_mine.append({
                            'x': well['x'],
                            'y': well['y'],
                            'Q': -Q,  # Negative for injection
                            'ton': ton,
                            'toff': toff,
                            'label': well['label'],
                                'type': 'Injection'
                            })
    
    # Display active wells summary
    if active_wells_mine:
        st.markdown("---")
        st.markdown("### 📋 Active Wells Summary")
        summary_wells = []
        for w in active_wells_mine:
            recovery_status = "Yes" if w['toff'] is not None else "No"
            toff_display = f"{w['toff']:.2f}" if w['toff'] is not None else "N/A"
            summary_wells.append({
                'Well': w['label'],
                'Type': w['type'],
                'Rate (m³/day)': f"{abs(w['Q']):.1f}",
                'Start (days)': f"{w['ton']:.2f}",
                'Stop (days)': toff_display,
                'Recovery': recovery_status
            })
        st.table(pd.DataFrame(summary_wells))
    
    # Analysis mode selection
    st.markdown("---")
    st.subheader("📊 Analysis Mode")
    analysis_mode = st.radio(
        "Select analysis type:",
        ["🗺️ Drawdown Map", "📈 Monitoring Well Time-Series", "🎯 Both"],
        horizontal=True
    )
    
    # Map parameters
    if analysis_mode in ["🗺️ Drawdown Map", "🎯 Both"]:
        st.markdown("---")
        st.markdown("### 🗺️ Map Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            map_time = st.number_input("Snapshot Time [days]", 
                                       value=365., min_value=365., max_value=36500.0, step=365.,
                                       help="Time for drawdown map snapshot")
        with col2:
            map_resolution = st.slider("Map Resolution", 
                                       min_value=30, max_value=100, value=50, step=10,
                                       help="Number of grid points (higher = slower)")
        with col3:
            contour_levels = st.slider("Contour Levels", 
                                       min_value=5, max_value=30, value=15, step=5,
                                       help="Number of contour lines")
        
        # Check if any wells are in recovery at map_time
        wells_in_recovery = [w for w in active_wells_mine if w['toff'] is not None and w['toff'] < map_time]
        if wells_in_recovery:
            st.info(f"ℹ️ At t={map_time:.1f} days, {len(wells_in_recovery)} well(s) will be in recovery phase")
    
    # Monitoring well parameters
    if analysis_mode in ["📈 Monitoring Well Time-Series", "🎯 Both"]:
        st.markdown("---")
        st.markdown("### 📈 Time-Series Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            ts_end_time = st.number_input("End Time [days]", 
                                          value=365.0, min_value=365., max_value=36500., step=365.0,
                                          help="Duration of monitoring period")
            ts_n_times = st.number_input("Number of Time Steps", 
                                         value=100, min_value=20, max_value=500, step=10)
        
        with col2:
            use_monitoring_wells = st.checkbox("Use placed monitoring wells", value=True)
            
            if use_monitoring_wells:
                monitoring_wells_list = st.session_state["wells_Monitoring"]
                if monitoring_wells_list:
                    selected_monitors = st.multiselect(
                        "Select monitoring wells:",
                        options=[f"{w['label']} ({w['x']:.0f}, {w['y']:.0f}) m" for w in monitoring_wells_list],
                        default=[f"{w['label']} ({w['x']:.0f}, {w['y']:.0f}) m" for w in monitoring_wells_list[:min(3, len(monitoring_wells_list))]]
                    )
                    monitor_locations = [w for w in monitoring_wells_list 
                                        if f"{w['label']} ({w['x']:.0f}, {w['y']:.0f}) m" in selected_monitors]
                else:
                    st.warning("⚠️ No monitoring wells placed")
                    monitor_locations = []
            else:
                st.info("Manual monitoring point entry:")
                n_monitors = st.number_input("Number of monitoring points", 
                                            min_value=1, max_value=10, value=3, step=1)
                monitor_locations = []
                cols = st.columns(min(3, n_monitors))
                for i in range(n_monitors):
                    with cols[i % 3]:
                        x_mon = st.number_input(f"Point {i+1} X [m]", value=500.0, step=50.0, key=f"mon_x_{i}")
                        y_mon = st.number_input(f"Point {i+1} Y [m]", value=500.0, step=50.0, key=f"mon_y_{i}")
                        monitor_locations.append({
                            'x': x_mon,
                            'y': y_mon,
                            'label': f"Mon{i+1}"
                        })
    
    # Run analysis
    st.markdown("---")
    if st.button("🚀 Run Dewatering Analysis", type="primary"):
        if not active_wells_mine:
            st.error("⚠️ No active wells configured!")
        else:
            with st.spinner('🔄 Computing dewatering predictions...'):
                # Create well objects with toff support
                well_objects_mine = []
                for aw in active_wells_mine:
                    if "Unconfined" in aquifer_type_mine:
                        well_obj = Neuman_well(
                            x=aw['x'], y=aw['y'], Q=aw['Q'], ton=aw['ton'],
                            T=mine_T, S=mine_S, b=mine_b, anis=mine_anis, 
                            Sy=mine_Sy, rwell=mine_rwell
                        )
                        if aw['toff'] is not None:
                            st.warning(f"⚠️ Recovery for well {aw['label']} not supported with Neuman solution")
                    else:
                        well_obj = Theis_well(
                            x=aw['x'], y=aw['y'], Q=aw['Q'], ton=aw['ton'],
                            T=mine_T, S=mine_S, rwell=mine_rwell,
                            toff=aw.get('toff', None)
                        )
                    well_objects_mine.append((well_obj, aw))  # Store well data with object
                
                results_dict = {}
                
                # ===== DRAWDOWN MAP CALCULATION =====
                if analysis_mode in ["🗺️ Drawdown Map", "🎯 Both"]:
                    st.info(f"🗺️ Computing drawdown map at t = {map_time} days...")
                    
                    # Use same map extent as tab 1 for consistency
                    x_min, x_max = 0, 10000
                    y_min, y_max = 0, 6000
                    
                    # Create grid
                    x_grid = np.linspace(x_min, x_max, map_resolution)
                    y_grid = np.linspace(y_min, y_max, map_resolution)
                    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                    
                    # Calculate drawdown at each grid point
                    Z_drawdown = np.zeros_like(X_grid)
                    
                    progress_bar = st.progress(0)
                    total_points = map_resolution * map_resolution
                    point_count = 0
                    
                    for i in range(map_resolution):
                        for j in range(map_resolution):
                            x_pt = X_grid[i, j]
                            y_pt = Y_grid[i, j]
                            
                            total_dd = 0.0
                            for well_obj, well_data in well_objects_mine:
                                total_dd += well_obj.s(x_pt, y_pt, map_time)
                            
                            Z_drawdown[i, j] = total_dd
                            point_count += 1
                            
                            if point_count % 100 == 0:
                                progress_bar.progress(point_count / total_points)
                    
                    progress_bar.empty()
                    
                    results_dict['map'] = {
                        'X': X_grid,
                        'Y': Y_grid,
                        'Z': Z_drawdown,
                        'time': map_time,
                        'extent': [x_min, x_max, y_min, y_max]
                    }
                
                # ===== TIME-SERIES CALCULATION =====
                if analysis_mode in ["📈 Monitoring Well Time-Series", "🎯 Both"]:
                    if monitor_locations:
                        st.info(f"📈 Computing time-series for {len(monitor_locations)} monitoring locations...")
                        
                        time_array = np.logspace(np.log10(0.01), np.log10(ts_end_time), ts_n_times)
                        
                        ts_results = {}
                        progress_bar = st.progress(0)
                        
                        for idx, mon in enumerate(monitor_locations):
                            drawdown_over_time = []
                            
                            for t in time_array:
                                total_dd = 0.0
                                for well_obj, well_data in well_objects_mine:
                                    total_dd += well_obj.s(mon['x'], mon['y'], t)
                                drawdown_over_time.append(total_dd)
                            
                            ts_results[mon['label']] = {
                                'time': time_array,
                                'drawdown': drawdown_over_time,
                                'x': mon['x'],
                                'y': mon['y']
                            }
                            
                            progress_bar.progress((idx + 1) / len(monitor_locations))
                        
                        progress_bar.empty()
                        results_dict['timeseries'] = ts_results
                
                # Store results
                st.session_state['mine_dewater_results'] = results_dict
                st.session_state['mine_active_wells'] = active_wells_mine
                
            st.success("✅ Analysis complete!")
    
    # ===== DISPLAY RESULTS =====
    if 'mine_dewater_results' in st.session_state:
        results = st.session_state['mine_dewater_results']
        active_wells_display = st.session_state['mine_active_wells']
        
        st.markdown("---")
        st.header("📊 Results")
        
        # ===== DRAWDOWN MAP VISUALIZATION =====
        if 'map' in results:
            st.subheader(f"🗺️ Drawdown Map at t = {results['map']['time']:.1f} days")
            
            # Show which wells are active/recovery at this time
            map_time_display = results['map']['time']
            pumping_wells_at_time = [w for w in active_wells_display 
                                     if w['ton'] <= map_time_display and 
                                     (w['toff'] is None or w['toff'] > map_time_display)]
            recovery_wells_at_time = [w for w in active_wells_display 
                                      if w['toff'] is not None and 
                                      w['ton'] <= map_time_display <= w['toff']]
            stopped_wells_at_time = [w for w in active_wells_display 
                                     if w['toff'] is not None and w['toff'] <= map_time_display]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Wells Pumping", len(pumping_wells_at_time))
            with col2:
                st.metric("Wells in Recovery", len(stopped_wells_at_time))
            with col3:
                st.metric("Total Active", len(active_wells_display))
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 11))
            
            X = results['map']['X']
            Y = results['map']['Y']
            Z = results['map']['Z']
            
            # Contour plot
            contour_filled = ax.contourf(X, Y, Z, levels=contour_levels, cmap='RdYlBu_r', alpha=0.7)
            contour_lines = ax.contour(X, Y, Z, levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f m')
            
            # Add colorbar
            cbar = plt.colorbar(contour_filled, ax=ax, label='Drawdown (m)', pad=0.02)
            
            # Plot wells with status
            plotted_pumping = False
            plotted_recovery = False
            plotted_injection = False
            
            for well in active_wells_display:
                # Determine well status at map time
                is_pumping = well['ton'] <= map_time_display and (well['toff'] is None or well['toff'] > map_time_display)
                is_recovery = well['toff'] is not None and well['toff'] <= map_time_display
                
                if well['Q'] > 0:  # Pumping well
                    if is_recovery:
                        # Well in recovery
                        ax.plot(well['x'], well['y'], 'co', markersize=14, 
                               markeredgecolor='white', markeredgewidth=2.5,
                               label='Recovery' if not plotted_recovery else '')
                        plotted_recovery = True
                        status_text = f"{well['label']}\n(Recovery)"
                    elif is_pumping:
                        # Well actively pumping
                        ax.plot(well['x'], well['y'], 'ko', markersize=14, 
                               markeredgecolor='white', markeredgewidth=2.5,
                               label='Pumping' if not plotted_pumping else '')
                        plotted_pumping = True
                        status_text = f"{well['label']}\n(Active)"
                    else:
                        # Well not yet started
                        ax.plot(well['x'], well['y'], 'wo', markersize=12, 
                               markeredgecolor='black', markeredgewidth=2,
                               label='Not Started' if well == active_wells_display[0] else '')
                        status_text = f"{well['label']}\n(Inactive)"
                else:  # Injection well
                    if is_recovery:
                        ax.plot(well['x'], well['y'], 'm^', markersize=14, 
                               markeredgecolor='white', markeredgewidth=2.5,
                               label='Injection-Stopped' if not plotted_injection else '')
                        status_text = f"{well['label']}\n(Stopped)"
                    elif is_pumping:
                        ax.plot(well['x'], well['y'], 'g^', markersize=14, 
                               markeredgecolor='white', markeredgewidth=2.5,
                               label='Injection' if not plotted_injection else '')
                        plotted_injection = True
                        status_text = f"{well['label']}\n(Active)"
                    else:
                        ax.plot(well['x'], well['y'], 'w^', markersize=12, 
                               markeredgecolor='black', markeredgewidth=2,
                               label='Injection-Inactive' if well == active_wells_display[0] else '')
                        status_text = f"{well['label']}\n(Inactive)"
                
                ax.text(well['x'], well['y']+50, status_text, 
                       ha='center', fontsize=9, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
            
            # Plot monitoring wells if in timeseries results
            if 'timeseries' in results:
                for mon_label, mon_data in results['timeseries'].items():
                    ax.plot(mon_data['x'], mon_data['y'], 'rs', markersize=11,
                           markeredgecolor='white', markeredgewidth=2, alpha=0.9,
                           label='Monitoring' if mon_label == list(results['timeseries'].keys())[0] else '')
                    ax.text(mon_data['x']+60, mon_data['y'], mon_label,
                           ha='left', fontsize=10, color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Plot polygons if they exist
            if st.session_state.polygons:
                for poly in st.session_state.polygons:
                    points = poly['points_meter']
                    polygon = MPLPolygon(points, fill=False, edgecolor='purple', 
                                        linewidth=2.5, linestyle='--', 
                                        label='Defined Area' if poly == st.session_state.polygons[0] else '')
                    ax.add_patch(polygon)
                    # Add label
                    centroid_x = np.mean([p[0] for p in points])
                    centroid_y = np.mean([p[1] for p in points])
                    ax.text(centroid_x, centroid_y, poly['name'],
                           ha='center', fontsize=12, color='purple', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, 
                                   edgecolor='purple', linewidth=2))
            
            ax.set_xlabel('X (m)', fontsize=13)
            ax.set_ylabel('Y (m)', fontsize=13)
            ax.set_title(f'Drawdown Contour Map - t = {results["map"]["time"]:.1f} days\n' + 
                        f'({len(pumping_wells_at_time)} pumping, {len(stopped_wells_at_time)} in recovery)', 
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Map statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Drawdown", f"{np.max(Z):.2f} m")
            with col2:
                st.metric("Min Drawdown", f"{np.min(Z):.2f} m")
            with col3:
                st.metric("Mean Drawdown", f"{np.mean(Z):.2f} m")
            with col4:
                st.metric("Std Dev", f"{np.std(Z):.2f} m")
        
        # ===== TIME-SERIES VISUALIZATION =====
        if 'timeseries' in results:
            st.markdown("---")
            st.subheader("📈 Drawdown Time-Series at Monitoring Locations")
            
            ts_data = results['timeseries']
            
            # Show recovery events on timeline
            recovery_events = [w for w in active_wells_display if w['toff'] is not None]
            if recovery_events:
                st.info(f"ℹ️ Recovery events: " + ", ".join([f"{w['label']} stops at t={w['toff']:.1f}d" for w in recovery_events]))
            
            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Linear scale
            ax = axes[0]
            for mon_label, mon_data in ts_data.items():
                # Calculate distance from nearest pumping well
                distances = []
                for well in active_wells_display:
                    if well['Q'] > 0:  # Only pumping wells
                        dist = np.sqrt((mon_data['x'] - well['x'])**2 + (mon_data['y'] - well['y'])**2)
                        distances.append(dist)
                min_dist = min(distances) if distances else 0
                
                ax.plot(mon_data['time'], mon_data['drawdown'], 'o-', 
                       markersize=4, label=f"{mon_label} (r={min_dist:.0f}m)", alpha=0.7, linewidth=2)
            
            # Add vertical lines for recovery events
            for well in recovery_events:
                if well['toff'] is not None:
                    ax.axvline(x=well['toff'], color='red', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax.text(well['toff'], ax.get_ylim()[1]*0.95, f"{well['label']}\nstops", 
                           rotation=90, va='top', ha='right', fontsize=8, color='red')
            
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel('Drawdown (m)', fontsize=12)
            ax.set_title('Drawdown vs Time (Linear Scale)', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Log-log scale
            ax = axes[1]
            for mon_label, mon_data in ts_data.items():
                distances = []
                for well in active_wells_display:
                    if well['Q'] > 0:
                        dist = np.sqrt((mon_data['x'] - well['x'])**2 + (mon_data['y'] - well['y'])**2)
                        distances.append(dist)
                min_dist = min(distances) if distances else 0
                
                # Filter positive drawdowns for log scale
                time_arr = np.array(mon_data['time'])
                dd_arr = np.array(mon_data['drawdown'])
                positive_mask = dd_arr > 0
                
                ax.loglog(time_arr[positive_mask], dd_arr[positive_mask], 'o-', 
                         markersize=4, label=f"{mon_label} (r={min_dist:.0f}m)", alpha=0.7, linewidth=2)
            
            # Add vertical lines for recovery events
            for well in recovery_events:
                if well['toff'] is not None:
                    ax.axvline(x=well['toff'], color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel('Drawdown (m)', fontsize=12)
            ax.set_title('Drawdown vs Time (Log-Log Scale)', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, which='both', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recovery analysis if applicable
            if recovery_events:
                st.markdown("---")
                st.markdown("### 🔄 Recovery Analysis")
                
                selected_recovery_well = st.selectbox(
                    "Select well for detailed recovery analysis:",
                    options=[w['label'] for w in recovery_events],
                    key="recovery_analysis_well"
                )
                
                recovery_well_data = [w for w in recovery_events if w['label'] == selected_recovery_well][0]
                toff_recovery = recovery_well_data['toff']
                ton_recovery = recovery_well_data['ton']
                Q_recovery = abs(recovery_well_data['Q'])
                
                st.info(f"📊 Analyzing recovery for **{selected_recovery_well}**: Pumped from t={ton_recovery:.2f} to t={toff_recovery:.2f} days at Q={Q_recovery:.0f} m³/day")
                
                # Create recovery plots for each monitoring location
                fig_rec, axes_rec = plt.subplots(1, 2, figsize=(16, 6))
                
                for mon_label, mon_data in ts_data.items():
                    t_array = np.array(mon_data['time'])
                    s_array = np.array(mon_data['drawdown'])
                    
                    # Filter for recovery period
                    recovery_mask = t_array >= toff_recovery
                    t_recovery = t_array[recovery_mask]
                    s_recovery = s_array[recovery_mask]
                    
                    if len(t_recovery) > 2:
                        # Plot 1: Residual drawdown vs time
                        ax = axes_rec[0]
                        t_prime = t_recovery - toff_recovery
                        ax.semilogx(t_prime, s_recovery, 'o-', markersize=5, label=mon_label, alpha=0.7)
                        
                        # Plot 2: Residual drawdown vs t/t'
                        ax = axes_rec[1]
                        t_over_tprime = t_recovery / t_prime
                        valid_mask = np.isfinite(t_over_tprime) & np.isfinite(s_recovery) & (s_recovery > 0)
                        if np.sum(valid_mask) > 0:
                            ax.semilogx(t_over_tprime[valid_mask], s_recovery[valid_mask], 'o-', 
                                       markersize=5, label=mon_label, alpha=0.7)
                
                axes_rec[0].set_xlabel("t' = t - t_off (days)", fontsize=12)
                axes_rec[0].set_ylabel('Residual Drawdown (m)', fontsize=12)
                axes_rec[0].set_title('Recovery: Drawdown vs Time Since Pump Stop', fontsize=13, fontweight='bold')
                axes_rec[0].legend(loc='best')
                axes_rec[0].grid(True, alpha=0.3)
                
                axes_rec[1].set_xlabel("t / t' (dimensionless)", fontsize=12)
                axes_rec[1].set_ylabel('Residual Drawdown (m)', fontsize=12)
                axes_rec[1].set_title('Theis Recovery Analysis (for T estimation)', fontsize=13, fontweight='bold')
                axes_rec[1].legend(loc='best')
                axes_rec[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_rec)
                
                # Estimate T from recovery
                st.markdown("#### 📐 Transmissivity Estimation from Recovery")
                col1, col2 = st.columns(2)
                
                with col1:
                    estimate_mon_well = st.selectbox(
                        "Select monitoring well for T estimation:",
                        options=list(ts_data.keys()),
                        key="estimate_mon_recovery"
                    )
                
                with col2:
                    # Calculate T
                    mon_data_est = ts_data[estimate_mon_well]
                    t_array = np.array(mon_data_est['time'])
                    s_array = np.array(mon_data_est['drawdown'])
                    
                    recovery_mask = t_array >= toff_recovery
                    t_recovery = t_array[recovery_mask]
                    s_recovery = s_array[recovery_mask]
                    
                    if len(t_recovery) > 3:
                        t_prime = t_recovery - toff_recovery
                        t_over_tprime = t_recovery / t_prime
                        log_t_ratio = np.log10(t_over_tprime)
                        
                        valid_mask = np.isfinite(log_t_ratio) & np.isfinite(s_recovery) & (s_recovery > 0)
                        
                        if np.sum(valid_mask) > 2:
                            coeffs = np.polyfit(log_t_ratio[valid_mask], s_recovery[valid_mask], 1)
                            slope_recovery = coeffs[0]
                            
                            T_recovery = (2.3 * Q_recovery) / (4 * np.pi * slope_recovery)
                            
                            st.metric("Estimated T", f"{T_recovery:.1f} m²/day")
                            st.metric("Recovery Slope", f"{slope_recovery:.4f} m/log-cycle")
                            
                            # Compare to input value
                            error_T = abs(T_recovery - mine_T) / mine_T * 100
                            st.metric("Error vs Input T", f"{error_T:.1f}%")
                        else:
                            st.warning("⚠️ Insufficient recovery data for T estimation")
                    else:
                        st.warning("⚠️ Need more time steps in recovery period")
            
            # Time-series data table
            with st.expander("📋 View Time-Series Data", expanded=False):
                # Create combined dataframe
                df_ts = pd.DataFrame()
                df_ts['Time (days)'] = list(ts_data.values())[0]['time']
                
                for mon_label, mon_data in ts_data.items():
                    df_ts[f'{mon_label} (m)'] = mon_data['drawdown']
                
                st.dataframe(df_ts.style.format("{:.4f}"), use_container_width=True)
                
                # Download button
                csv_ts = df_ts.to_csv(index=False)
                st.download_button(
                    "💾 Download Time-Series Data (CSV)",
                    data=csv_ts,
                    file_name=f"timeseries_dewatering_{st.session_state.student_id}.csv",
                    mime="text/csv"
                )
            
            # Summary statistics for each monitoring location
            st.markdown("### 📊 Monitoring Location Summary")
            summary_data = []
            for mon_label, mon_data in ts_data.items():
                final_dd = mon_data['drawdown'][-1]
                max_dd = max(mon_data['drawdown'])
                max_time = mon_data['time'][np.argmax(mon_data['drawdown'])]
                
                # Find nearest pumping well
                min_dist = float('inf')
                nearest_well = 'N/A'
                for well in active_wells_display:
                    if well['Q'] > 0:
                        dist = np.sqrt((mon_data['x'] - well['x'])**2 + (mon_data['y'] - well['y'])**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_well = well['label']
                
                summary_data.append({
                    'Location': mon_label,
                    'Coordinates': f"({mon_data['x']:.0f}, {mon_data['y']:.0f})",
                    'Nearest Well': nearest_well,
                    'Distance (m)': f"{min_dist:.1f}",
                    'Final Drawdown (m)': f"{final_dd:.3f}",
                    'Max Drawdown (m)': f"{max_dd:.3f}",
                    'Time to Max (days)': f"{max_time:.2f}"
                })
            
            st.table(pd.DataFrame(summary_data))

# ============================================================================
# TAB 5: HELP
# ============================================================================
with tab5:
    st.header("📚 Backgroundand and instructions")
    
    st.markdown("""
    ### 🎯 Purpose
    This simulator helps you understand groundwater flow to wells. This is not designed to be an exact replication of a real-world scenario but rather help understand the concepts of well hydraulics appropriate to pumping tests and dewatering. The exercise unitises superposition of analytical solutions to represent multiple pumps during dewatering.
    
    ### 🔧 How to Use
    1. **Add Wells & Areas Tab**: Place pumping/monitoring wells and injection wells
    2. **Pumping Test Tab**: Configure test parameters and run simulation to generate data, and analuys the data to determine aquifer properties.
    3. **Area Analysis Tab**: Evaluate average drawdown in defined polygonal areas
    4. **Mine Dewatering Tab**: Predict drawdown from multiple wells
    
    ### 📐 Exercise 1
    1. Place a pumping well close to the mine site (e.g., within 200m)
    2. Add a monitoring well ~10m from the pumping well.
    3. Configure a pumping test (e.g., Q=4320 m³/day for 10 days) and run the simulation.
    4. Analyse the time-series data to estimate T and S. Move the sliders to match the values and focus on replicating the late time data.
    5. Repeat steps 1-4 with two more combinations of pumping and monitoring well locations. Try running tests for 20 days and 50 days.
    6. Compare your estimated T and S values from each test.
    7. Use the save session feature in the left hang menu (download the file). This will be used in exercise 2.
                
    ### 📐 Exercise 2:
    1. Load your saved session from exercise 1.
    2. Use the mine dewatering tab. Staring with your existing wells.
    3. use the figures of the target water level vs preducted water level to observe how the existing wells preform in dewatering the mine sites.
    4. Add additional wells as needed to achieve the target water levels.
    5. Consider the recovery by utilising the "Include recovery Tab" for each well.
    6. Try to minimise the number of wells and pumping rates while achieving the target water levels.
    
    ### 📐 Exercise 3:
    1. Explore the impact of dewatering on the regional groundwater levels using the regional impacts tab.
    2. Observe how drawdown propagates over time and distance from the mine site, including at the GDE.
    3. Add a mionitoring well at the GDE location to observe the time-series drawdown there.
    4. Consider strategies to mitigate impacts on the GDE while still achieving mine dewatering (placement of wells, utilisation of injection wells).
    
    """)
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
        'T': np.random.uniform(100, 2000),
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Add Wells & Areas", "🔬 Pumping Test", "📊 Area Analysis", "⛏️ Mine Dewatering", "📚 Help & Theory"])

# ============================================================================
# TAB 1: ADD WELLS AND AREAS - WITH PROPER UPDATING
# ============================================================================
with tab1:
    st.header("Well Placement & Area Definition")
    
    # Initialize canvas refresh counter if not exists
    if 'canvas_refresh' not in st.session_state:
        st.session_state.canvas_refresh = 0
    
    # Load basemap
    try:
        original_image = Image.open("Training_base_map.png")
        target_width = 1000
        scale = target_width / original_image.width
        scaled_height = int(original_image.height * scale)
        resized_image = original_image.resize((target_width, scaled_height))
    except FileNotFoundError:
        st.warning("Base map not found. Using blank canvas.")
        resized_image = Image.new('RGB', (1000, 600), color='lightgray')
        scaled_height = 600
        target_width = 1000

    # Drawing mode selection
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        drawing_mode = st.selectbox("Drawing Mode", ["Wells", "Areas (Rectangle)"])
    
    if drawing_mode == "Wells":
        with col2:
            well_type = st.selectbox("Select Well Type", ["Pumping", "Injection", "Monitoring"])
        with col3:
            st.info(f"💡 Draw circles to place {well_type.lower()} wells. Each pixel = {PIXEL_TO_METER} meters.")
    else:  # Rectangle mode
        with col2:
            polygon_name = st.text_input("Area Name", value=f"Area_{len(st.session_state.polygons)+1}", key="rect_name_input")
        with col3:
            st.info(f"💡 Click and drag to draw rectangles. They save automatically!")

    # Clear and Refresh buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Clear All Wells"):
            for key in ["wells_Pumping", "wells_Injection", "wells_Monitoring"]:
                st.session_state[key] = []
            st.session_state.canvas_refresh += 1
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Areas"):
            st.session_state.polygons = []
            st.session_state.canvas_refresh += 1
            st.rerun()
    with col3:
        # Manual refresh button for when canvas doesn't update
        if st.button("🔄 Refresh Canvas", help="Click if your drawings don't appear"):
            st.session_state.canvas_refresh += 1
            st.rerun()
    
    # Colors
    type_colors = {
        "Pumping": "rgba(0, 0, 255, 0.6)",
        "Injection": "rgba(0, 128, 0, 0.6)",
        "Monitoring": "rgba(255, 165, 0, 1.0)",
        "Rectangle": "rgba(255, 0, 255, 0.3)"
    }
    
    # Canvas configuration
    if drawing_mode == "Wells":
        selected_color = type_colors[well_type]
        canvas_drawing_mode = "circle"
    else:  # Rectangle mode
        selected_color = type_colors["Rectangle"]
        canvas_drawing_mode = "rect"
    
    type_keys = {
        "Pumping": "wells_Pumping",
        "Injection": "wells_Injection",
        "Monitoring": "wells_Monitoring"
    }
    type_prefix = {"Pumping": "P", "Injection": "I", "Monitoring": "M"}

    # Prepare initial drawing with existing wells
    initial_drawing = {"objects": []}
    
    # Add existing wells
    for label, key in type_keys.items():
        for well in st.session_state[key]:
            color = type_colors[well["type"]]
            x_pixel = well.get("x_pixel", well["x"] / PIXEL_TO_METER)
            y_pixel = well.get("y_pixel", well["y"] / PIXEL_TO_METER)
            initial_drawing["objects"].append({
                "type": "circle",
                "left": x_pixel - 5,
                "top": y_pixel - 5,
                "radius": 5,
                "fill": color
            })
            initial_drawing["objects"].append({
                "type": "text",
                "left": x_pixel + 8,
                "top": y_pixel - 5,
                "text": well["label"],
                "font": "Arial",
                "fontSize": 12,
                "fill": color
            })
    
    # Add existing rectangles (stored as polygons with 4 vertices)
    for poly in st.session_state.polygons:
        points = poly['points_pixel']
        if len(points) == 4:  # Only show rectangles
            # Calculate rectangle bounds
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            left = min(x_coords)
            top = min(y_coords)
            width = max(x_coords) - left
            height = max(y_coords) - top
            
            initial_drawing["objects"].append({
                "type": "rect",
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "fill": "rgba(255, 0, 255, 0.2)",
                "stroke": "rgba(255, 0, 255, 1.0)",
                "strokeWidth": 2
            })
            
            # Add label at center
            center_x = left + width / 2
            center_y = top + height / 2
            initial_drawing["objects"].append({
                "type": "text",
                "left": center_x - 20,
                "top": center_y - 7,
                "text": poly["name"],
                "font": "Arial",
                "fontSize": 14,
                "fill": "rgba(255, 0, 255, 1.0)",
                "fontWeight": "bold"
            })

    # Canvas with dynamic key to force updates
    canvas_key = f"canvas_{st.session_state.student_id}_{st.session_state.canvas_refresh}"
    
    canvas_result = st_canvas(
        fill_color=selected_color,
        stroke_width=2,
        stroke_color="rgba(0, 0, 0, 0)",
        background_image=resized_image,
        update_streamlit=True,
        height=scaled_height,
        width=target_width,
        drawing_mode=canvas_drawing_mode,
        initial_drawing=initial_drawing,
        key=canvas_key,  # Dynamic key forces refresh
    )
    
    # Instruction text
    st.caption("💡 **Tip:** After drawing, click the **🔄 Refresh Canvas** button above if your drawing doesn't save automatically.")

    # Process canvas objects
    if canvas_result.json_data is not None:
        canvas_objects = canvas_result.json_data["objects"]
        
        if drawing_mode == "Wells":
            # Process wells
            canvas_circles = [obj for obj in canvas_objects if obj["type"] == "circle"]
            current_canvas_count = len(canvas_circles)
            total_stored = sum(len(st.session_state[key]) for key in type_keys.values())
            
            if current_canvas_count > total_stored:
                # Get all stored well coordinates
                all_stored_coords = set()
                for key in type_keys.values():
                    all_stored_coords.update({
                        (round(w.get("x_pixel", w["x"] / PIXEL_TO_METER), 1), 
                         round(w.get("y_pixel", w["y"] / PIXEL_TO_METER), 1)) 
                        for w in st.session_state[key]
                    })

                # Add new wells
                for obj in canvas_circles:
                    x_pixel = round(obj["left"] + obj["radius"], 1)
                    y_pixel = round(obj["top"] + obj["radius"], 1)
                    x_meter = x_pixel * PIXEL_TO_METER
                    y_meter = y_pixel * PIXEL_TO_METER
                    
                    if (x_pixel, y_pixel) not in all_stored_coords:
                        prefix = type_prefix[well_type]
                        count = len(st.session_state[type_keys[well_type]])
                        label = f"{prefix}{count + 1}"
                        
                        st.session_state[type_keys[well_type]].append({
                            "x": x_meter,
                            "y": y_meter,
                            "x_pixel": x_pixel,
                            "y_pixel": y_pixel,
                            "label": label,
                            "type": well_type
                        })
                        st.session_state.canvas_refresh += 1
                        st.success(f"✅ Added {well_type} well: {label}")
                        st.rerun()
        
        elif drawing_mode == "Areas (Rectangle)":
            # Process rectangles
            canvas_rectangles = [obj for obj in canvas_objects if obj["type"] == "rect"]
            
            if len(canvas_rectangles) > 0:
                rect_obj = canvas_rectangles[-1]
                left = rect_obj.get("left", 0)
                top = rect_obj.get("top", 0)
                width = rect_obj.get("width", 0)
                height = rect_obj.get("height", 0)
                
                if width > 5 and height > 5:
                    # Rectangle corners as polygon
                    points_pixel = [
                        (left, top),
                        (left + width, top),
                        (left + width, top + height),
                        (left, top + height)
                    ]
                    
                    # Check if this rectangle is new
                    is_new = True
                    for stored_poly in st.session_state.polygons:
                        if len(stored_poly['points_pixel']) == 4:
                            first_stored = stored_poly['points_pixel'][0]
                            first_new = points_pixel[0]
                            if abs(first_stored[0] - first_new[0]) < 5 and abs(first_stored[1] - first_new[1]) < 5:
                                stored_width = stored_poly['points_pixel'][1][0] - stored_poly['points_pixel'][0][0]
                                stored_height = stored_poly['points_pixel'][2][1] - stored_poly['points_pixel'][0][1]
                                if abs(stored_width - width) < 5 and abs(stored_height - height) < 5:
                                    is_new = False
                                    break
                    
                    if is_new:
                        points_meter = [(x * PIXEL_TO_METER, y * PIXEL_TO_METER) for x, y in points_pixel]
                        area_m2 = width * height * (PIXEL_TO_METER ** 2)
                        
                        st.session_state.polygons.append({
                            "name": polygon_name,
                            "points_pixel": points_pixel,
                            "points_meter": points_meter,
                            "area_m2": area_m2
                        })
                        st.session_state.canvas_refresh += 1
                        st.success(f"✅ Saved rectangle: {polygon_name} ({area_m2:.0f} m²)")
                        st.rerun()

    # Display summary
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Placed Wells Summary")
        total_wells = sum(len(st.session_state[key]) for key in type_keys.values())
        st.metric("Total Wells", total_wells)
        
        cols = st.columns(3)
        for idx, (label, key) in enumerate(type_keys.items()):
            with cols[idx]:
                st.write(f"**{label}** ({len(st.session_state[key])})")
                if st.session_state[key]:
                    for well in st.session_state[key]:
                        st.caption(f"{well['label']}: ({well['x']:.0f}, {well['y']:.0f}) m")
                else:
                    st.caption(f"None")
    
    with col2:
        st.subheader("📐 Defined Areas")
        st.metric("Total Areas", len(st.session_state.polygons))
        
        if st.session_state.polygons:
            for i, poly in enumerate(st.session_state.polygons):
                with st.expander(f"**{poly['name']}**", expanded=False):
                    st.metric("Area", f"{poly['area_m2']:.0f} m²")
                    st.caption(f"Type: Rectangle")
                    if st.button(f"🗑️ Remove", key=f"remove_poly_{i}"):
                        st.session_state.polygons.pop(i)
                        st.session_state.canvas_refresh += 1
                        st.rerun()
        else:
            st.caption("No areas defined yet")

# ============================================================================
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
                init_S = 1e-4
                init_Q = Q
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fit_T = st.slider(
                    "Transmissivity (T) [m²/day]",
                    min_value=10.0,
                    max_value=5000.0,
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
    st.header("📊 Drawdown Analysis for Defined Areas")
    
    if not st.session_state.polygons:
        st.warning("⚠️ No areas defined yet. Go to 'Add Wells & Areas' tab to draw polygons.")
    else:
        st.info("💡 Evaluate drawdown evolution within defined areas from pumping/injection wells (with optional recovery).")
        
        # Aquifer parameters
        st.subheader("🌊 Analysis Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            area_T = st.number_input("Transmissivity (T) [m²/day]", 
                                     value=500.0, min_value=1.0, max_value=10000.0, step=10.0,
                                     key="area_T")
        with col2:
            area_S = st.number_input("Storativity (S)", 
                                     value=0.0001, min_value=1e-6, max_value=0.5, 
                                     format="%.6f", step=1e-5,
                                     key="area_S")
        with col3:
            area_rwell = st.number_input("Well Radius [m]", 
                                         value=0.15, min_value=0.01, max_value=1.0, step=0.01,
                                         key="area_rwell")
        
        # Time-series parameters
        st.subheader("⏱️ Time-Series Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            area_end_time = st.number_input("End Time [days]", 
                                            value=365.0, min_value=0.1, max_value=7300.0, step=10.0,
                                            help="Duration of analysis period")
        with col2:
            area_n_times = st.number_input("Number of Time Steps", 
                                           value=100, min_value=20, max_value=500, step=10,
                                           help="More points = smoother curves but slower")
        with col3:
            grid_density = st.slider("Grid Density", min_value=10, max_value=50, value=20, step=5,
                                     help="Number of grid points per area dimension")
        
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
                                                   value=1000.0, min_value=0.0, step=100.0,
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
                    
                    # Create time array
                    time_array = np.logspace(np.log10(0.01), np.log10(area_end_time), area_n_times)
                    
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
            st.subheader("📈 Drawdown Evolution Over Time")
            
            # Plot options
            col1, col2 = st.columns(2)
            with col1:
                plot_scale = st.radio("Time axis scale:", ["Linear", "Log"], horizontal=True, index=1)
            with col2:
                show_recovery_lines = st.checkbox("Show recovery events", value=True)
            
            # Get recovery events
            recovery_events = [w for w in analysis_wells if w.get('toff') is not None]
            
            # Create plots - one per area
            for area_data in area_results:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                time = area_data['time']
                
                # Plot min, mean, max as filled area
                ax.fill_between(time, area_data['min_drawdown'], area_data['max_drawdown'], 
                               alpha=0.2, color='steelblue', label='Min-Max Range')
                ax.plot(time, area_data['mean_drawdown'], '-', linewidth=2.5, color='darkblue', 
                       label='Mean Drawdown')
                ax.plot(time, area_data['min_drawdown'], '--', linewidth=1.5, color='steelblue', 
                       alpha=0.7, label='Minimum')
                ax.plot(time, area_data['max_drawdown'], '--', linewidth=1.5, color='navy', 
                       alpha=0.7, label='Maximum')
                
                # Add recovery event lines
                if show_recovery_lines and recovery_events:
                    for well in recovery_events:
                        if well.get('toff') is not None:
                            ax.axvline(x=well['toff'], color='red', linestyle='--', 
                                     linewidth=1.5, alpha=0.5)
                            ax.text(well['toff'], ax.get_ylim()[1]*0.95, 
                                   f"{well['label']}\nstops", 
                                   rotation=90, va='top', ha='right', fontsize=8, color='red',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                ax.set_xlabel('Time (days)', fontsize=12)
                ax.set_ylabel('Drawdown (m)', fontsize=12)
                ax.set_title(f"Area: {area_data['name']} ({area_data['area_m2']:.0f} m², {area_data['n_points']} sample points)", 
                           fontsize=13, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3, which='both' if plot_scale == 'Log' else 'major')
                
                if plot_scale == 'Log':
                    ax.set_xscale('log')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Comparison plot - all areas on one plot
            st.markdown("---")
            st.subheader("📊 Area Comparison")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(area_results)))
            
            for idx, area_data in enumerate(area_results):
                time = area_data['time']
                ax.plot(time, area_data['mean_drawdown'], '-', linewidth=2.5, 
                       color=colors[idx], label=f"{area_data['name']} (Mean)", alpha=0.8)
                ax.fill_between(time, area_data['min_drawdown'], area_data['max_drawdown'], 
                               alpha=0.15, color=colors[idx])
            
            # Add recovery lines
            if show_recovery_lines and recovery_events:
                for well in recovery_events:
                    if well.get('toff') is not None:
                        ax.axvline(x=well['toff'], color='red', linestyle='--', 
                                 linewidth=1.5, alpha=0.5)
            
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel('Mean Drawdown (m)', fontsize=12)
            ax.set_title('Comparison of Mean Drawdown Across All Areas', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, which='both' if plot_scale == 'Log' else 'major')
            
            if plot_scale == 'Log':
                ax.set_xscale('log')
            
            plt.tight_layout()
            st.pyplot(fig)
            
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
                        'Min_Drawdown_m': area_data['min_drawdown'][i],
                        'Mean_Drawdown_m': area_data['mean_drawdown'][i],
                        'Max_Drawdown_m': area_data['max_drawdown'][i]
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
    st.header("⛏️ Mine Dewatering Analysis")
    
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
                                 value=0.0001, min_value=1e-6, max_value=0.5, 
                                 format="%.6f", step=1e-5,
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
    
    use_placed_wells = st.checkbox("Use wells from 'Add Wells & Areas' tab", value=True, key="use_placed_mine")
    
    active_wells_mine = []
    
    if use_placed_wells:
        pumping_wells_list = st.session_state["wells_Pumping"]
        injection_wells_list = st.session_state["wells_Injection"]
        
        if not pumping_wells_list and not injection_wells_list:
            st.warning("⚠️ No wells placed. Add wells in the first tab or uncheck the box above.")
        else:
            st.success(f"Found {len(pumping_wells_list)} pumping and {len(injection_wells_list)} injection wells")
            
            # Pumping wells with recovery option
            if pumping_wells_list:
                st.markdown("**⚙️ Pumping Wells:**")
                
                # Create expandable sections for each well
                for idx, well in enumerate(pumping_wells_list):
                    with st.expander(f"**{well['label']}** - ({well['x']:.0f}, {well['y']:.0f}) m", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            Q = st.number_input(f"Rate [m³/day]", 
                                               value=1000.0, min_value=0.0, step=100.0,
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
                            include_recovery = st.checkbox(f"Include Stop", value=False, 
                                                          key=f"mine_recovery_{well['label']}",
                                                          help="Simulate when this well stops injecting")
                        with col4:
                            active = st.checkbox(f"Active", value=True, key=f"mine_active_{well['label']}")
                        
                        if include_recovery:
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
                                       value=100.0, min_value=0.01, max_value=3650.0, step=10.0,
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
                                          value=365.0, min_value=0.1, max_value=7300.0, step=10.0,
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
                    
                    # Determine map extent from well locations
                    all_x = [w['x'] for w in active_wells_mine]
                    all_y = [w['y'] for w in active_wells_mine]
                    
                    x_min, x_max = min(all_x) - 500, max(all_x) + 500
                    y_min, y_max = min(all_y) - 500, max(all_y) + 500
                    
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
    st.header("📚 Help & Theory")
    
    st.markdown("""
    ### 🎯 Purpose
    This simulator helps you understand groundwater flow during pumping tests.
    
    ### 🔧 How to Use
    1. **Add Wells & Areas Tab**: Place pumping/monitoring wells and draw areas of interest
    2. **Pumping Test Tab**: Configure test parameters and run simulation
    3. **Area Analysis Tab**: Evaluate average drawdown in defined polygonal areas
    4. **Mine Dewatering Tab**: Predict drawdown from multiple wells
    
    ### 📐 Theory
    
    **Theis Solution (Confined Aquifer)**
    - Assumes: Homogeneous, isotropic, confined aquifer
    - Key parameters: T (transmissivity), S (storativity)
    
    **Neuman Solution (Unconfined Aquifer)**
    - Accounts for delayed yield from storage
    - Key parameters: T, S, Sy (specific yield), anisotropy
    
    ### 📊 Area Analysis
    - Define areas using polygon drawing tool
    - Calculate average drawdown within each area
    - Compare impacts across different regions
    - Useful for environmental impact assessment
    
    ### 🧮 Analysis Tasks
    1. **Cooper-Jacob Method**: Use semi-log plot to estimate T and S
    2. **Curve Matching**: Match observed data to type curves
    3. **Distance-Drawdown**: Analyze drawdown vs distance at fixed time
    4. **Area Impact**: Evaluate drawdown in specific areas
    
    ### 💡 Tips
    - Place monitoring wells at various distances
    - Define areas around sensitive features (wetlands, rivers, etc.)
    - Longer pumping tests provide more data
    - Compare confined vs unconfined solutions
    """)
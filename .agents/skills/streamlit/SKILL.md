---
name: streamlit
description: >-
  Official Streamlit development skill for building interactive web applications,
  dashboards, session state management, UI layout optimization, Matplotlib/Plotly visualizations,
  and resource caching.
---

# 🎨 Official Streamlit Development Skill

This skill provides best practices, architectural patterns, and code snippets for building fast, beautiful, and interactive web applications using **Streamlit**.

---

## 🚀 1. Application Lifecycle & Entry Point

- **Page Configuration**: Always initialize `st.set_page_config` as the very first Streamlit command.
  ```python
  import streamlit as st

  st.set_page_config(
      page_title="App Title",
      page_icon="🧬",
      layout="wide",
      initial_sidebar_state="expanded"
  )
  ```

- **Running the App**:
  ```powershell
  python -m streamlit run app/main.py
  ```

---

## ⚡ 2. Caching & Performance Optimization

Streamlit provides two primary decorators for caching:

### `@st.cache_resource` (For Global Objects & Connections)
Use `@st.cache_resource` for un-picklable, heavy, or global objects such as machine learning models, database connections, and API clients.

```python
@st.cache_resource
def load_ml_model(model_path: str):
    import joblib
    return joblib.load(model_path)
```

### `@st.cache_data` (For Computations & DataFrames)
Use `@st.cache_data` for functions that return data structures (DataFrames, dicts, lists, numbers).

```python
@st.cache_data
def fetch_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Preprocessing logic
    return df
```

---

## 📐 3. UI Layout & Component Organization

- **Columns & Grids**:
  ```python
  col1, col2, col3 = st.columns([3, 2, 1])
  with col1:
      st.subheader("Primary Section")
  with col2:
      st.metric("Metric Name", "Value", delta="+5%")
  ```

- **Tabs**:
  ```python
  tab_raw, tab_api = st.tabs(["📝 Input Sequence", "🌐 API Accession"])
  with tab_raw:
      seq_input = st.text_area("Sequence Input:")
  ```

- **Sidebar**:
  ```python
  st.sidebar.header("⚙️ Configuration")
  host_selection = st.sidebar.selectbox("Select Host Organism", ["E. coli K-12", "E. coli BL21"])
  ```

---

## 🔄 4. Session State & Interactivity

Keep state persistent across reruns using `st.session_state`:

```python
if "counter" not in st.session_state:
    st.session_state.counter = 0

def increment_counter():
    st.session_state.counter += 1

st.button("Increment", on_click=increment_counter)
st.write(f"Current Count: {st.session_state.counter}")
```

---

## 📊 5. Render Matplotlib & Pyplot Visualizations

Ensure clean figure creation and resource management when embedding Matplotlib/Pyplot charts:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot([1, 2, 3], [4, 5, 6])
st.pyplot(fig, clear_figure=True)
plt.close(fig)
```

---

## 🚨 6. Error Handling & User Banners

Inform users of warnings, errors, and success states using built-in alert banners:

```python
st.success("✅ Execution successful!")
st.info("ℹ️ Running in fallback heuristic mode.")
st.warning("⚠️ Warning: Parameter boundary exceeded.")
st.error("❌ Failed to process construct sequence.")
```

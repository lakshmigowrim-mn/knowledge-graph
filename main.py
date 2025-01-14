import pandas as pd
import streamlit as st

from neo4j import GraphDatabase
from neo4j import Query
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt


from utils import create_the_knowledge_graph


st.markdown(
"""
# How to Run `neo4j` locally:

## Using Docker Container:
 - `docker pull neo4j:latest`
 - `docker run -d --name neo4j-container -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`
 
### If you are running on WSL use the wsl IP instead of localhost:
- `ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d/ -f1`
"""
)


class HRMKnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver("neo4j://172.30.211.20:7687", auth=("neo4j", "password"))
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    def close(self):
        self.driver.close()

    def add_relationship(self, source, relationship, target):
        query = """
        MERGE (s:Entity {name: $source})
        MERGE (t:Entity {name: $target})
        MERGE (s)-[r:`""" + relationship + """`]->(t)
        """
        with self.driver.session() as session:
            session.run(query, source=source, target=target)

    def get_relationships(self, entity):
        query = Query("""
        MATCH (s:Entity {name: $entity})-[r]->(t)
        RETURN s.name AS source, type(r) AS relationship, t.name AS target
        """)
        with self.driver.session() as session:
            result = session.run(query, entity=entity)
            return [record for record in result]

    def get_all_relationships(self):
        query = """
        MATCH (s)-[r]->(t)
        RETURN s.name AS source, type(r) AS relationship, t.name AS target
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]

    @staticmethod
    def visualize_graph_matplotlib(relationships):
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges to the graph
        for rel in relationships:
            head = rel["head"]
            tail = rel["tail"]
            relation_type = rel["type"]

            G.add_node(head)
            G.add_node(tail)
            G.add_edge(head, tail, label=relation_type)

        # Draw the graph
        pos = nx.spring_layout(G)  # Spring layout for positioning nodes
        fig, ax = plt.subplots(figsize=(20, 10))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='skyblue',
            node_size=2000,
            font_size=10,
            font_color='black',
            font_weight='bold',
            edge_color='gray',
            arrowsize=20,
            arrowstyle='-|>'
        )

        # Draw edge labels (relationship types)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        return fig

    def fire_dashboard(self):
        if "relationships" not in st.session_state:
            st.session_state["relationships"] = None
        if "extraction_done" not in st.session_state:
            st.session_state["extraction_done"] = False
        if "network" not in st.session_state:
            st.session_state["network"] = None

        st.title("Knowledge Graph for Human Resource Management")

        # Manual addition of relationships
        st.subheader("Add Relationships from Text")

        content_input = st.text_area("Enter text containing relationships", height=200)

        if st.button("Extract and Add Relationships"):
            with st.spinner("Running and extracting relationships...."):
                if content_input:
                    # Use the function to extract relationships
                    relationships = create_the_knowledge_graph(transformers_tokenizer=self.tokenizer,
                                                               transformers_model=self.model,
                                                               content_input=content_input,
                                                               show_verbose=False)
                    st.session_state["relationships"] = relationships

                    # Add each extracted relationship to the graph
                    for rel in relationships:
                        head = rel["head"]
                        relation_type = rel["type"]
                        tail = rel["tail"]
                        self.add_relationship(head, relation_type, tail)

                    st.success(f"{len(relationships)} relationships extracted and added to the graph.")

                    # Optionally, display the extracted relationships
                    st.write("Extracted Relationships:")
                    fig = self.visualize_graph_matplotlib(relationships)
                    st.session_state["network"] = fig
                    st.pyplot(fig, use_container_width=False)

                    st.session_state["extraction_done"] = True
                else:
                    st.error("Please enter some text to extract relationships.")

        # Query the graph
        relationships = st.session_state["relationships"]
        if st.session_state["extraction_done"]:
            st.subheader("Query the Graph ... ")
            query_entity = st.text_input("Enter an Entity to Query")

            if st.button("Query Relationships"):
                if query_entity:
                    results = self.get_relationships(query_entity)
                    results_df = pd.DataFrame(results, columns=["source", "relationship", "target"])
                    if results:
                        st.pyplot(st.session_state["network"], use_container_width=False)
                        st.write(f"Relationships for '{query_entity}':")
                        st.table(results_df)
                    else:
                        st.error(f"Entity '{query_entity}' not found in the graph.")
                else:
                    st.error("Please enter an entity to query.")

        # # Upload dataset and batch input
        # with upload_tab:
        #     st.subheader("Upload a Dataset")
        #
        #     uploaded_file = st.file_uploader("Upload CSV/JSON File", type=["csv", "json"])
        #
        #     if uploaded_file:
        #         if uploaded_file.name.endswith("csv"):
        #             data = pd.read_csv(uploaded_file)
        #         elif uploaded_file.name.endswith("json"):
        #             data = pd.read_json(uploaded_file)
        #
        #         st.write("Preview of Uploaded Data:", data.head())
        #
        #         source_col = st.selectbox("Select Source Column", data.columns)
        #         relationship_col = st.selectbox("Select Relationship Column", data.columns)
        #         target_col = st.selectbox("Select Target Column", data.columns)
        #
        #         if st.button("Add Relationships from File"):
        #             for _, row in data.iterrows():
        #                 self.add_relationship(row[source_col], row[relationship_col], row[target_col])
        #             st.success("Relationships from file added to the graph.")


# Usage example (replace with your actual Neo4j connection details)
if __name__ == "__main__":
    app = HRMKnowledgeGraph()
    app.fire_dashboard()
    app.close()
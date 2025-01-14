import spacy
import networkx as nx
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
import wikipedia
import IPython
from pyvis.network import Network
from networkx.drawing import nx_agraph
import subprocess
import base64
import string


# Install graphviz from Conda
# conda install -c conda-forge pygraphviz


def normalize_text(text):
    """
    Lowercases text, strips whitespace, optionally removes punctuation.
    You can expand this function for more sophisticated normalization.
    """
    text = text.strip().lower()
    # Optional: remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def find_similar_entity(new_entity, existing_entities):
    """
    Check if 'new_entity' is a substring or a superstring
    of any existing entity (in a normalized sense).
    If found, return the existing entity name to unify them.
    Otherwise, return None.
    """
    norm_new = normalize_text(new_entity)

    for e in existing_entities:
        norm_existing = normalize_text(e)
        # If either is a substring of the other, consider them the same
        if norm_new in norm_existing or norm_existing in norm_new:
            return e
    return None


def interpret_model_output(model_text):
    """
    Parse the REBEL model output to extract relations in
    the form of (subject, relation, object).
    """
    relation_list = []
    subject_str, relation_str, object_str = '', '', ''
    current_field = ''

    # Remove special tokens that we don't need
    cleaned_output = (model_text
                      .replace("<s>", "")
                      .replace("<pad>", "")
                      .replace("</s>", "")
                      .strip())

    for token in cleaned_output.split():
        if token == "<triplet>":
            current_field = 'SUBJECT'
            if relation_str != '':
                relation_list.append({
                    'head': subject_str.strip(),
                    'type': relation_str.strip(),
                    'tail': object_str.strip()
                })
                relation_str = ''
            subject_str = ''

        elif token == "<subj>":
            current_field = 'OBJECT'
            if relation_str != '':
                relation_list.append({
                    'head': subject_str.strip(),
                    'type': relation_str.strip(),
                    'tail': object_str.strip()
                })
            object_str = ''

        elif token == "<obj>":
            current_field = 'RELATION'
            relation_str = ''

        else:
            if current_field == 'SUBJECT':
                subject_str += ' ' + token
            elif current_field == 'OBJECT':
                object_str += ' ' + token
            elif current_field == 'RELATION':
                relation_str += ' ' + token

    # Catch any final leftover relation
    if subject_str != '' and relation_str != '' and object_str != '':
        relation_list.append({
            'head': subject_str.strip(),
            'type': relation_str.strip(),
            'tail': object_str.strip()
        })

    return relation_list


class KnowledgeBaseManager:
    def __init__(self):
        self.all_relations = []
        self.known_entities = set()  # Keep track of all entity labels

    def are_triples_same(self, triple_a, triple_b):
        """
        Checks if two relations have the exact same triple:
        subject (head), relation (type), object (tail).
        """
        return (triple_a["head"] == triple_b["head"]
                and triple_a["type"] == triple_b["type"]
                and triple_a["tail"] == triple_b["tail"])

    def is_relation_present(self, triple_data):
        """
        Determine if a relation already exists in the knowledge base.
        """
        for existing_triple in self.all_relations:
            if self.are_triples_same(triple_data, existing_triple):
                return True
        return False

    def combine_relation_data(self, new_triple):
        """
        If a relation already exists, merge any extra metadata (like spans).
        """
        for stored_triple in self.all_relations:
            if self.are_triples_same(new_triple, stored_triple):
                # Merge new spans that are not already present.
                new_spans = [
                    s for s in new_triple["meta"]["spans"]
                    if s not in stored_triple["meta"]["spans"]
                ]
                stored_triple["meta"]["spans"] += new_spans
                return

    def unify_entity_label(self, label):
        """
        Attempt to unify an entity label with known entities 
        using approximate or substring matching. If found, return the known label.
        Otherwise, add and return the new label.
        """
        # Step 1: Check if there's a near match in known_entities
        matched_label = find_similar_entity(label, self.known_entities)
        if matched_label is not None:
            # Return the existing label
            return matched_label
        else:
            # If not found, add label to known_entities
            self.known_entities.add(label)
            return label

    def insert_relation(self, triple_data):
        """
        Adds a relation to the knowledge base if it does not exist.
        If it exists, merges the new metadata.
        """
        # Normalize HEAD
        unified_head = self.unify_entity_label(triple_data["head"])
        # Normalize TAIL
        unified_tail = self.unify_entity_label(triple_data["tail"])

        triple_data["head"] = unified_head
        triple_data["tail"] = unified_tail

        if not self.is_relation_present(triple_data):
            self.all_relations.append(triple_data)
        else:
            self.combine_relation_data(triple_data)

    def display_relations(self):
        """
        Prints all relations in the knowledge base.
        """
        print("Knowledge Base Relations:")
        for idx, rel_item in enumerate(self.all_relations, start=1):
            print(f"{idx:02d}. {rel_item}")

    def get_relations(self):
        """
        Returns the entire list of relations stored.
        """
        return self.all_relations


def build_knowledgebase_from_text(transformers_tokenizer, transformers_model, input_text, span_limit=128,
                                  show_verbose=False):
    """
    Splits input text (or a single sentence) into overlapping spans, feeds each span
    into the REBEL model to extract relations, and compiles them
    into a KnowledgeBaseManager object.
    """
    # Tokenize the entire text
    tokenized_input = transformers_tokenizer([input_text], return_tensors="pt")

    # Calculate total tokens
    total_tokens = len(tokenized_input["input_ids"][0])
    if show_verbose:
        print(f"Total tokens in input: {total_tokens}")

    # Calculate how many spans are needed
    total_spans = math.ceil(total_tokens / span_limit)
    if show_verbose:
        print(f"Total spans needed: {total_spans}")

    # Overlap calculation for the spans
    overlap_count = math.ceil(
        (total_spans * span_limit - total_tokens) /
        max(total_spans - 1, 1)
    )

    # Determine span boundaries
    boundary_indices = []
    start_pos = 0
    for i in range(total_spans):
        boundary_indices.append([
            start_pos + span_limit * i,
            start_pos + span_limit * (i + 1)
        ])
        start_pos -= overlap_count

    if show_verbose:
        print("Span boundaries determined:")
        for boundary in boundary_indices:
            print(boundary)

    # Split the tokens/mask into spans
    chunked_input_ids = [
        tokenized_input["input_ids"][0][b[0]:b[1]]
        for b in boundary_indices
    ]
    chunked_attention_masks = [
        tokenized_input["attention_mask"][0][b[0]:b[1]]
        for b in boundary_indices
    ]

    # Create an expanded input dictionary
    stacked_inputs = {
        "input_ids": torch.stack(chunked_input_ids),
        "attention_mask": torch.stack(chunked_attention_masks)
    }

    # Generation config
    generation_params = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3
    }

    # Generate outputs for each span
    model_generated_outputs = transformers_model.generate(
        **stacked_inputs,
        **generation_params
    )

    # Decode the outputs
    decoded_model_texts = transformers_tokenizer.batch_decode(
        model_generated_outputs,
        skip_special_tokens=False
    )

    # Create a knowledge base instance
    kb_instance = KnowledgeBaseManager()

    for idx, decoded_text in enumerate(decoded_model_texts):
        # Determine which span this result came from
        current_span_index = idx // generation_params["num_return_sequences"]
        extracted_relations = interpret_model_output(decoded_text)

        for rel_obj in extracted_relations:
            rel_obj["meta"] = {
                "spans": [boundary_indices[current_span_index]]
            }
            kb_instance.insert_relation(rel_obj)

    return kb_instance


def build_kb_from_sentences(transformers_tokenizer, transformers_model, input_text, show_verbose=False):
    """
    Use spaCy (or another approach) to split the text into sentences.
    Then process each sentence separately to ensure we capture all
    relations, and combine the results into one KnowledgeBaseManager.
    """
    # Load spaCy's small English model (ensure installed: python -m spacy download en_core_web_sm)
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(input_text)
    sentence_list = [sent.text.strip() for sent in doc.sents]

    if show_verbose:
        print("Splitting text into sentences:")
        for s in sentence_list:
            print(f"- {s}")

    # Create a single KB manager to combine all sentence-level KBs
    combined_kb = KnowledgeBaseManager()

    for sent_index, sentence_text in enumerate(sentence_list):
        # Build a temporary KB from one sentence
        temp_kb = build_knowledgebase_from_text(
            transformers_tokenizer,
            transformers_model,
            sentence_text,
            span_limit=64,  # smaller spans since sentences are short
            show_verbose=show_verbose
        )

        # Merge the temporary KB's relations into the combined KB
        for r in temp_kb.get_relations():
            combined_kb.insert_relation(r)

    return combined_kb


def create_the_knowledge_graph(transformers_tokenizer, transformers_model, content_input,
                               show_verbose=True,
                               # output_file_name="generated_knowledge_graph.html"
                               ):
    # Instead of passing the entire multi-sentence text directly,
    # we now split it by sentences and parse each sentence.
    final_kb = build_kb_from_sentences(transformers_tokenizer, transformers_model, content_input, show_verbose=False)
    # final_kb.display_relations()

    # Build a directed graph from the combined relations
    # graph_directed = nx.DiGraph()
    found_relations = final_kb.get_relations()
    return found_relations


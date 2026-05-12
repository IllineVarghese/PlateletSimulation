GRN Engine

This module implements the Gene Regulatory Network (GRN) controlling platelet behavior.

Modules:

graphml_parser.py
    Loads GRN topology from GraphML files.

grn_model.py
    Stores network structure and metadata.

grn_state.py
    Stores node values for each agent.

grn_stepper.py
    Executes time-step updates for the GRN.

io_mapping.py
    Maps simulation sensors to GRN inputs and GRN outputs to agent behavior.
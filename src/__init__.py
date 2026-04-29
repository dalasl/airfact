"""
Terminal Data Leakage Detection System (DLP-Profiling)

Three-layer architecture:
    src.ch2_user_profiling    -- Perception layer (profiling, clustering)
    src.ch3_sensitive_grading -- Cognition layer  (RAG grading, self-consistency)
    src.ch4_rule_generation   -- Execution layer   (rule gen, adaptive decision)

Cross-module services:
    src.server                -- DLPServer orchestrator
    src.event_bus             -- Publish-subscribe event bus
    src.terminal_agent        -- Endpoint agent (file/process watcher)
    src.velociraptor_bridge   -- gRPC bridge to Velociraptor VQL engine
"""

__version__ = "1.0.0"

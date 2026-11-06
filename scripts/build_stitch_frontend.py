"""
NutriMatch AI Pot — Frontend Build Script
=========================================
Compiles data/stitch_template.html into src/frontend/static/index.html.
Ensures:
1. Complete Brand Identity: NutriMatch AI Pot
2. Simplified Header with Logo, Search, Saved Count, Profile, and Online Indicator
3. Compact, Collapsible Sidebar with 5 Main Destinations:
   - Discover Recipes
   - Saved Recipes
   - My Taste Profile
   - Model Evaluation
   - Research & Diagnostics
4. Discover Recipes page with real-time Knowledge-Based Filters and Taste Profile card
5. Compact, Attractive Recipe Cards with Real Backend Data and 5-Star Online SGD Feedback
6. Saved Recipes view with Empty State and Collection Management
7. My Taste Profile with Dynamic Palate Gauges and Recent History from live API
8. Dedicated Model Evaluation with live benchmark results from models_saved/benchmark_results.json
9. Research & Diagnostics workspace with Latent Taste Space, Ingredient Affinities, Bias-Variance curves, and SGD simulation
"""

import os
import sys

def build_frontend():
    template_path = os.path.join(os.path.dirname(__file__), "..", "data", "stitch_template.html")
    template_path = os.path.abspath(template_path)
    output_path = os.path.join(os.path.dirname(__file__), "..", "src", "frontend", "static", "index.html")
    output_path = os.path.abspath(output_path)

    if not os.path.exists(template_path):
        print(f"Error: Template file not found at {template_path}")
        sys.exit(1)

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Validate essential elements
    required_ids = [
        'sidebar-toggle-btn', 'app-sidebar', 'main-content-wrapper',
        'header-search-input', 'saved-counter-badge', 'user-select',
        'recipe-cards-container', 'load-more-btn', 'tab-panel-discover',
        'tab-panel-saved', 'tab-panel-profile', 'tab-panel-benchmarks',
        'tab-panel-research', 'latent-canvas', 'ingredient-canvas'
    ]

    missing = [req_id for req_id in required_ids if f'id="{req_id}"' not in html]
    if missing:
        print(f"Warning: Expected element IDs missing in template: {missing}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Successfully generated {output_path} ({len(html):,} bytes)")

if __name__ == "__main__":
    build_frontend()

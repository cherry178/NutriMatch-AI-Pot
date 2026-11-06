"""
Unit Tests for Data Ingestion, Preprocessing, and Splitting
===========================================================
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import (
    safe_parse_list,
    parse_iso_duration,
    derive_dietary_flags,
    export_recbole_atomic_files
)
from src.data.splitter import stratified_user_split


def test_safe_parse_list():
    s = 'c("blueberries", "granulated sugar", "vanilla yogurt")'
    parsed = safe_parse_list(s)
    assert parsed == ["blueberries", "granulated sugar", "vanilla yogurt"]

    empty_parsed = safe_parse_list(None)
    assert empty_parsed == []


def test_parse_iso_duration():
    assert parse_iso_duration("PT25M") == 25
    assert parse_iso_duration("PT1H30M") == 90
    assert parse_iso_duration("PT2H") == 120
    assert parse_iso_duration(None) == 0


def test_derive_dietary_flags():
    veg_ing = ["spinach", "ricotta", "pasta", "olive oil"]
    veg_tags = ["italian", "vegetarian"]
    flags = derive_dietary_flags(veg_ing, veg_tags)
    assert flags["is_vegetarian"] is True
    assert flags["is_vegan"] is False  # contains ricotta
    assert flags["is_gluten_free"] is False  # contains pasta

    vegan_ing = ["tofu", "broccoli", "rice", "soy sauce"]
    vegan_tags = ["asian", "vegan"]
    flags_v = derive_dietary_flags(vegan_ing, vegan_tags)
    assert flags_v["is_vegetarian"] is True
    assert flags_v["is_vegan"] is True

    meat_ing = ["chicken", "garlic", "rice"]
    flags_m = derive_dietary_flags(meat_ing, [])
    assert flags_m["is_vegetarian"] is False


def test_recbole_atomic_export(tmp_path):
    df_inter = pd.DataFrame({
        "user_id": [1, 1, 2],
        "recipe_id": [101, 102, 101],
        "rating": [5.0, 4.0, 3.0],
        "date": ["2021-01-01", "2021-01-02", "2021-01-03"]
    })
    df_rec = pd.DataFrame({
        "RecipeId": [101, 102],
        "Name": ["Pasta Primavera", "Berry Smoothie"],
        "RecipeCategory": ["Dinner", "Beverage"],
        "Calories": [350.0, 180.0]
    })

    out_dir = str(tmp_path / "recbole")
    inter_f, item_f = export_recbole_atomic_files(df_inter, df_rec, output_dir=out_dir)

    assert os.path.exists(inter_f)
    assert os.path.exists(item_f)

    # Check header conformity to RecBole format
    inter_lines = open(inter_f).readlines()
    assert "user_id:token\trecipe_id:token\trating:float\ttimestamp:float\n" == inter_lines[0]

    item_lines = open(item_f).readlines()
    assert "recipe_id:token\tname:token_seq\tcategory:token\tcalories:float\n" == item_lines[0]


def test_stratified_user_split():
    df = pd.DataFrame({
        "user_id": [1] * 10 + [2] * 10,
        "recipe_id": list(range(10)) + list(range(10)),
        "rating": [5.0] * 20,
        "date": [f"2021-01-{i+1:02d}" for i in range(10)] * 2
    })
    train, val, test = stratified_user_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(train) + len(val) + len(test) == 20
    assert set(train["user_id"]) == {1, 2}
    assert set(val["user_id"]) == {1, 2}
    assert set(test["user_id"]) == {1, 2}

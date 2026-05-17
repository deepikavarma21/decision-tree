# TODO - Streamlit Decision Tree Regressor

- [x] Refactor `app.py` to train on `laptop_price.csv` (remove synthetic dataset usage).

- [ ] Add robust preprocessing for `Ram` and `Weight` columns (clean numeric strings).

- [ ] Fit a `Pipeline` with `ColumnTransformer` for numeric/categorical features.
- [ ] Implement model persistence with `decision_tree_model.pkl` (load if exists, otherwise train + save).
- [ ] Keep sidebar inputs, prediction, metrics, feature importance, and decision tree visualization.
- [ ] Run `streamlit run app.py` to validate the app starts.

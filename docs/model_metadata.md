# Model Metadata

- Nama di Model Registry: debit-sungai-regressor
- Versi: 1
- Run: rf_depth_3 dengan ID e1ae72b033644ffa96e1229f818b8138
- Algoritma: Random Forest Regressor
- Hyperparameter: n_estimators=200, max_depth=3, min_samples_leaf=1
- Data: dataset_20261011_084241.csv, tag Git data-v2, hash DVC 9aaa854aa1171ad3b495e64e7f29fd6d.dir
- Pembagian data: berurutan waktu, train 70%, validasi 15%, test 15%
- Fitur: 10 fitur dari dataset, target river_discharge
- Metrik validasi: RMSE 13.898, MAE 9.093, NSE 0.586, RMSE debit tinggi 38.165
- Metrik test: RMSE 8.298, MAE 5.619, NSE 0.476
- Alasan pemilihan: Memiliki val_rmse terendah dibanding model lain, mengalahkan baseline, dan selisih train_rmse dengan val_rmse tidak terlalu jauh sehingga tidak overfit. Model ini juga dipilih karena waktu training paling cepat dibanding GBR yang performanya hampir seri.
- Memuat model: mlflow.sklearn.load_model("models:/debit-sungai-regressor@champion")
- Catatan: model disimpan dengan cloudpickle, muat hanya dari sumber tepercaya
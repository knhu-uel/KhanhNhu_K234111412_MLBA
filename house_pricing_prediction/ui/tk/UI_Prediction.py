import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from house_price_prediction.ui.tk.DatasetViewer import DatasetViewer
from house_price_prediction.ui.tk.Predictor import save_model, load_model


FEATURES = [
    'Avg Area Income',
    'Avg Area House Age',
    'Avg Area Number of Rooms',
    'Avg Area Number of Bedrooms',
    'Area Population',
]

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ZIP = str(MODEL_DIR / 'house_price_model.zip')


class UIPrediction(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('House Price Prediction — Tkinter')
        # Tăng kích thước để khớp layout như ảnh minh họa
        self.geometry('1200x700')
        self.df = None
        self.model = None
        self._build_ui()

    def _build_ui(self):
        # Header
        header = ttk.Label(self, text='House Pricing Prediction', anchor='center', font=('Arial', 16, 'bold'))
        header.pack(fill=tk.X, padx=8, pady=6)

        # Top toolbar: dataset, training rate, actions
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        # Select dataset display
        ttk.Label(top, text='Select Dataset:').pack(side=tk.LEFT, padx=(0,4))
        self.dataset_path_var = tk.StringVar(value='(none)')
        ttk.Label(top, textvariable=self.dataset_path_var, width=60).pack(side=tk.LEFT, padx=(0,10))

        # Training rate (percent)
        ttk.Label(top, text='Training Rate:').pack(side=tk.LEFT)
        self.train_rate_var = tk.StringVar(value='80')  # 80%
        ttk.Entry(top, textvariable=self.train_rate_var, width=6).pack(side=tk.LEFT, padx=(2,10))

        # Action buttons matching PDF numbering
        ttk.Button(top, text='1. Pick Dataset', command=self.pick_dataset).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='2. View Dataset', command=self.view_dataset).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='3. Train Model', command=self.train_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='4. Evaluate Model', command=self.evaluate_model).pack(side=tk.LEFT, padx=4)
        # Lưu mô hình chỉ nằm trong panel đánh giá (đúng ảnh)

        # Main area: left dataset viewer, right evaluation panel
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Đặt panel đánh giá bên phải, cố định chiều rộng để luôn hiển thị
        right = ttk.LabelFrame(main, text='Evaluation is finished')
        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8,0))
        right.configure(width=380)
        right.pack_propagate(False)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.viewer = DatasetViewer(left)
        self.viewer.pack(fill=tk.BOTH, expand=True)

        # Coefficient viewer + metrics in right panel
        ttk.Label(right, text='Coefficient').pack(anchor='w', padx=6, pady=(6,0))
        self.coeff_viewer = DatasetViewer(right)
        self.coeff_viewer.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.lbl_intercept = ttk.Label(right, text='Intercept: -')
        self.lbl_intercept.pack(anchor='w', padx=6, pady=(4,0))
        self.lbl_mae = ttk.Label(right, text='Mean Absolute Error (MAE): -')
        self.lbl_mae.pack(anchor='w', padx=6)
        self.lbl_mse = ttk.Label(right, text='Mean Square Error (MSE): -')
        self.lbl_mse.pack(anchor='w', padx=6)
        self.lbl_rmse = ttk.Label(right, text='Root Mean Square Error (RMSE): -')
        self.lbl_rmse.pack(anchor='w', padx=6)
        ttk.Button(right, text='5. Save Model', command=self.save_model_ui).pack(padx=6, pady=6)

        # Bottom area: Load Model + prediction inputs
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(bottom, text='6. Load Model').grid(row=0, column=0, sticky='w', padx=4, pady=4)
        self.model_choice = tk.StringVar(value=os.path.basename(MODEL_ZIP))
        self.model_choices = []
        self.model_menu = ttk.OptionMenu(bottom, self.model_choice, None)
        self.model_menu.grid(row=0, column=1, sticky='w', padx=4, pady=4)
        ttk.Button(bottom, text='Load Model', command=self.load_model_ui).grid(row=0, column=2, sticky='w', padx=4, pady=4)

        # Prediction inputs
        fields = FEATURES
        self.pred_inputs = {}
        for i, f in enumerate(fields, start=1):
            ttk.Label(bottom, text=f).grid(row=i, column=0, sticky='w', padx=4, pady=4)
            var = tk.StringVar()
            ttk.Entry(bottom, textvariable=var, width=20).grid(row=i, column=1, sticky='w', padx=4, pady=4)
            self.pred_inputs[f] = var
        ttk.Button(bottom, text='7. Prediction House Pricing', command=self.predict_inline).grid(row=len(fields)+1, column=0, sticky='w', padx=4, pady=6)
        self.pred_out_var = tk.StringVar(value='Prediction Price: -')
        ttk.Label(bottom, textvariable=self.pred_out_var).grid(row=len(fields)+1, column=1, sticky='w', padx=4, pady=6)

        # Status
        self.status = tk.StringVar(value='Ready')
        ttk.Label(self, textvariable=self.status).pack(fill=tk.X, padx=8, pady=6)

        # Initialize dropdown
        self._refresh_model_dropdown()

    # Actions
    def pick_dataset(self):
        path = filedialog.askopenfilename(title='Select dataset CSV', filetypes=[('CSV files', '*.csv')])
        if path:
            try:
                self.df = pd.read_csv(path)
                self.dataset_path_var.set(path)
                self.status.set(f'Loaded dataset: {os.path.basename(path)}')
            except Exception as e:
                messagebox.showerror('Error', str(e))

    def view_dataset(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        self.viewer.load_dataframe(self.df)

    def _get_X_y(self):
        X = self.df[FEATURES]
        y = self.df['Price']
        return X, y

    def train_model(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        X, y = self._get_X_y()
        try:
            rate_percent = int(self.train_rate_var.get())
            rate_percent = max(50, min(rate_percent, 95))  # clamp 50..95
        except Exception:
            rate_percent = 80
        rate = rate_percent / 100.0
        test_size = 1.0 - rate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        save_model(lin, FEATURES, MODEL_ZIP)
        self.model = lin
        messagebox.showinfo('Train', f'Trained LinearRegression (Training Rate={rate_percent}%). Saved: {MODEL_ZIP}')
        self._refresh_model_dropdown()

    def evaluate_model(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        try:
            self.model, _ = load_model(MODEL_ZIP)
        except Exception:
            self.status.set('Model not found. Train first.')
            return
        X, y = self._get_X_y()
        # sử dụng cùng tỉ lệ test theo Training Rate
        try:
            rate_percent = int(self.train_rate_var.get())
            rate_percent = max(50, min(rate_percent, 95))
        except Exception:
            rate_percent = 80
        test_size = 1.0 - (rate_percent/100.0)
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        # Cập nhật panel đánh giá và hiển thị coefficients
        coeff_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': self.model.coef_})
        self.coeff_viewer.load_dataframe(coeff_df)
        self.lbl_intercept.config(text=f'Intercept: {self.model.intercept_:.6f}')
        self.lbl_mae.config(text=f'Mean Absolute Error (MAE): {mae:.6f}')
        self.lbl_mse.config(text=f'Mean Square Error (MSE): {mse:.6f}')
        self.lbl_rmse.config(text=f'Root Mean Square Error (RMSE): {rmse:.6f}')

        # Hiển thị bảng dự đoán bên trái theo đúng layout
        df2 = self.df.copy()
        df2['Prediction'] = self.model.predict(df2[FEATURES])
        df_show = df2[[*FEATURES, 'Price', 'Prediction']].rename(columns={'Price': 'Original Price', 'Prediction': 'Prediction Price'})
        self.viewer.load_dataframe(df_show)

    def predict_inline(self):
        try:
            # ưu tiên model đang nạp; nếu chưa có thì nạp mặc định
            if not self.model:
                self.model, _ = load_model(MODEL_ZIP)
            x = [[float(self.pred_inputs[f].get()) for f in FEATURES]]
            pred = float(self.model.predict(x)[0])
            self.pred_out_var.set(f'Prediction Price: {pred:,.2f}')
        except Exception as e:
            messagebox.showerror('Error', f'Invalid input: {e}')

    def predict_by_dataset(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        try:
            self.model, _ = load_model(MODEL_ZIP)
        except Exception:
            self.status.set('Model not found. Train first.')
            return
        df2 = self.df.copy()
        df2['Prediction'] = self.model.predict(df2[FEATURES])
        self.viewer.load_dataframe(df2[[*FEATURES, 'Price', 'Prediction']])

    # Save/Load model UI helpers
    def save_model_ui(self):
        if not self.model:
            # Nếu chưa train, thử nạp mặc định
            try:
                self.model, _ = load_model(MODEL_ZIP)
            except Exception:
                messagebox.showinfo('Info', 'Train model before saving.')
                return
        if not messagebox.askyesno('Save Model', 'Lưu mô hình hiện tại với tên kèm thời gian?'):
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_zip = MODEL_DIR / f'house_price_model_{ts}.zip'
        save_model(self.model, FEATURES, str(out_zip))
        messagebox.showinfo('Saved', f'Lưu mô hình vào: {out_zip}')
        self._refresh_model_dropdown()

    def _refresh_model_dropdown(self):
        # Quét thư mục models để nạp danh sách zip
        zips = sorted([p.name for p in MODEL_DIR.glob('*.zip')])
        if not zips:
            zips = [os.path.basename(MODEL_ZIP)] if os.path.exists(MODEL_ZIP) else []
        self.model_choices = zips
        menu = self.model_menu['menu']
        menu.delete(0, 'end')
        for name in self.model_choices:
            menu.add_command(label=name, command=lambda n=name: self.model_choice.set(n))
        if zips:
            # đặt giá trị mặc định
            self.model_choice.set(zips[-1])

    def load_model_ui(self):
        name = self.model_choice.get()
        if not name:
            messagebox.showinfo('Info', 'Không tìm thấy file mô hình.')
            return
        zip_path = MODEL_DIR / name
        try:
            self.model, _ = load_model(str(zip_path))
            self.status.set(f'Loaded model: {name}')
        except Exception as e:
            messagebox.showerror('Error', f'Không thể nạp mô hình: {e}')


if __name__ == '__main__':
    app = UIPrediction()
    app.mainloop()
# 🏡 EstateIQ — AI House Price Predictor

A fully functional **Machine Learning powered house price prediction web app** that estimates property values in both **USD** and **INR**, built with Python (Flask) on the backend and a rich animated HTML/CSS/JS frontend.

---

## ✨ Features

- 🤖 **Advanced Gradient Boosting ML Model** — trained on 4,600+ real-world housing records
- 💵 **Dual Currency Output** — predictions in both USD ($) and INR (₹)
- 🎬 **Smash Entrance Animation** — cinematic splash screen with a smash-out transition
- 🌄 **Rotating Background Wallpapers** — dynamic real estate wallpapers that crossfade
- 🪄 **3D Tilt Form Cards** — interactive glassmorphism cards with Vanilla-Tilt.js
- ✨ **Special Popup Result** — a glowing cinematic popup displays the full prediction breakdown
- ⭐ **Reviews Section** — scrollable testimonials from real estate professionals
- 📖 **About / How the AI Works** — explains the Gradient Boosting model to users

---

## 🗂️ Project Structure

```
├── index.html        # Full frontend (HTML + CSS + JS, single file)
├── app.py            # Flask backend with /predict API endpoint
├── trainmodel.py     # Script to train and save the ML model
├── data.csv          # Housing dataset (4,600+ records)
├── requirements.txt  # Python dependencies
└── README.md
```

> ⚠️ `model.pkl` is excluded from Git (generated locally). See **Setup** below.

---

## 🚀 Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the ML model (generates `model.pkl`)
```bash
python trainmodel.py
```

### 4. Start the Flask server
```bash
python app.py
```

### 5. Open in browser
Go to → **http://127.0.0.1:5000**

---

## 🧠 ML Architecture

| Component | Detail |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Pipeline | SimpleImputer → StandardScaler → GBR |
| Features | sqft_living, bedrooms, bathrooms, floors, waterfront, view, condition, sqft_lot, yr_built |
| Dataset | 4,600+ King County, WA housing sales |
| Currency | USD with live-rate INR conversion (83x) |

---

## 📸 Screenshots

> Add screenshots here after uploading!

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

Built with ❤️ by **Vivek**

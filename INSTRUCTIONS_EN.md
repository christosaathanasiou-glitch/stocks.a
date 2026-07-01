# 🛠️ Stock Analyzer — Step-by-Step Instructions

This guide walks you through everything: installing, running locally, fixing common
errors, and putting the app online for free. No prior experience needed.

---

## Part 1 — Install Python (one time)

1. Check if you already have it. Open a terminal:
   - **Windows:** press the Windows key, type `cmd` or `powershell`, press Enter.
   - **Mac:** press Cmd+Space, type `terminal`, press Enter.
2. Type this and press Enter:
   ```
   python --version
   ```
   - If it shows something like `Python 3.11`, you're set — skip to Part 2.
   - If it says "not found", download Python from https://www.python.org/downloads/
     On Windows, **tick "Add Python to PATH"** in the installer.

---

## Part 2 — Get the files in one folder

Put these files together in a folder named `stock_analyzer`:
- `app_en.py` (the English app)
- `requirements.txt`

(You can keep `app.py` too if you also want the Greek version.)

---

## Part 3 — Open the terminal INSIDE that folder

This is the step people most often miss. Python must run *from inside* the folder
that contains `app_en.py`.

**Easy method:**
1. In the terminal, type `cd ` (the letters c, d, then a space) — don't press Enter yet.
2. Drag the `stock_analyzer` folder from your file explorer into the terminal window.
   The full path fills in automatically.
3. Now press Enter.

**Check you're in the right place:** type `dir` (Windows) or `ls` (Mac) and press Enter.
You should see `app_en.py` in the list. If you don't, you're in the wrong folder.

Your prompt should now end with `...\stock_analyzer>`.

---

## Part 4 — Install the dependencies (one time)

```
pip install -r requirements.txt
```

Wait for it to finish (it downloads Streamlit, yfinance, pandas, plotly).

---

## Part 5 — Run the app

```
streamlit run app_en.py
```

It opens in your browser at http://localhost:8501.

To stop it: go back to the terminal and press **Ctrl+C**.
To run it again later: repeat Parts 3 and 5.

---

## Common errors & fixes

**"streamlit : The term 'streamlit' is not recognized…"** (Windows)
Use this instead:
```
python -m streamlit run app_en.py
```

**"File does not exist: app_en.py"**
You're not inside the folder. Redo Part 3 (drag the folder after `cd`), confirm with
`dir`/`ls` that `app_en.py` shows up, then run again.

**"No data found for {SYMBOL}"** even though the symbol is correct
Yahoo Finance rate-limited you temporarily. Wait 30–60 seconds, then press the
**"Clear cache & retry"** button in the app, or restart it (Ctrl+C, then run again).

**`python` not recognized**
Try `py` instead, e.g. `py -m streamlit run app_en.py`.

---

## Part 6 — Put it online for free (optional)

This gives you a shareable link that works on any browser or phone.

1. Create a free account at https://github.com
2. Create a new repository (e.g. `stock-analyzer`), then **Add file → Upload files**
   and drag in `app_en.py` and `requirements.txt`. Click **Commit changes**.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **New app** → pick your repository → set the main file to `app_en.py` →
   **Deploy**.

In 1–2 minutes it's live at a `your-app.streamlit.app` address.

### Updating the online app later
Streamlit Cloud auto-updates whenever the file changes on GitHub:
1. On GitHub, open `app_en.py` → click the pencil (✏️ Edit).
2. Select all (Ctrl+A), delete, paste the new version, then **Commit changes**.
   (Or use **Add file → Upload files** to overwrite it.)
3. It refreshes automatically within ~2 minutes (or click **Reboot app** in the panel).

### Customizing the online app
- **Subdomain** (the `xxx` in `xxx.streamlit.app`): change it free under
  the app's **Settings → General**.
- **Hide the top-right Streamlit menu:** already done via CSS in the app.
- **Full custom domain** (e.g. `myapp.com`) and removing the "Hosted with Streamlit"
  badge are **not** available on the free tier — they need a paid plan or self-hosting
  (e.g. Render, Railway) with a domain you own.

---

## Reminder

This tool is for **education only** and is **not investment advice**. Indicators and
the Buy/Hold/Sell chart summarize current technical signals — they don't predict the
future. Markets are also moved by news and events no indicator can foresee.

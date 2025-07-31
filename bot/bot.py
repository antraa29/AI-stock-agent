import sys
import os
import discord
from discord.ext import commands
import joblib
import pandas as pd
from dotenv import load_dotenv

# Setup environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Load model & scaler
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(base_dir, 'models', 'model.pkl')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

try:
    print(f"🔍 Loading model from: {model_path}")
    print(f"🔍 Loading scaler from: {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"❌ Failed to load model or scaler: {e}")
    exit()

# Import feature function
try:
    from train_model.prepare_dataset import fetch_and_engineer_features
except ImportError as e:
    print(f"❌ Error importing feature function: {e}")
    exit()

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Prediction logic
def predict_from_features(features_df):
    scaled = scaler.transform(features_df)
    prediction = model.predict(scaled)
    return prediction[0]

@bot.event
async def on_ready():
    print(f"✅ Bot is online as {bot.user}!")

@bot.command(name='predict')
async def predict_stock(ctx, ticker: str):
    await ctx.send(f"📊 Fetching data for `{ticker.upper()}`...")

    try:
        df = fetch_and_engineer_features(ticker)
        latest = df.drop(columns=['Signal'], errors='ignore').tail(1)

        prediction = predict_from_features(latest)
        signal = "📈 **Buy Signal**" if prediction == 1 else "📉 **Sell Signal**"

        await ctx.send(f"✅ Prediction for `{ticker.upper()}`: {signal}")

    except ValueError as ve:
        await ctx.send(f"⚠️ Error: `{ve}`")
        print(f"❌ No data for {ticker.upper()}: {ve}")
    except Exception as e:
        await ctx.send(f"❌ Unexpected error during prediction.")
        print(f"❌ Error predicting for {ticker.upper()}: {e}")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN not set in .env file.")
    else:
        bot.run(TOKEN)

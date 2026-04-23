import pandas as pd

# LangGraph system
from core.graph import build_graph

# ML pipeline
from core.ml_pipeline import run_ml_pipeline, improve_model


def main():
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

    print("🚀 Running Multi-Agent System...\n")

    # Run LangGraph agents
    app = build_graph()
    result = app.invoke({"df": df})

    # Print agent outputs (optional, can comment if too much)
    for key, value in result.items():
        print(f"\n=== {key.upper()} ===")
        print(value)

    print("\n🚀 Running Real AutoML Pipeline...\n")

    # Run ML pipeline
    ml_result = run_ml_pipeline(df)

    print("\n=== MODEL COMPARISON ===")

    for model, metrics in ml_result["results"].items():
        print(f"\n{model}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    print(f"\n🏆 BEST MODEL: {ml_result['best_model']}")
    print("📊 Graph saved as model_comparison.png")

    # 🔁 USER FEEDBACK LOOP
    while True:
        user_input = input("\nIs this accuracy sufficient? (yes/no): ").lower()

        if user_input == "yes":
            print("✅ Final model accepted!")
            break

        elif user_input == "no":
            improved_acc = improve_model(df)

            best_acc = ml_result["results"][ml_result["best_model"]]["accuracy"]

            if improved_acc > best_acc:
                print("🔥 Model improved!")
            else:
                print("⚠️ No significant improvement.")

        else:
            print("Please type 'yes' or 'no'")


if __name__ == "__main__":
    main()
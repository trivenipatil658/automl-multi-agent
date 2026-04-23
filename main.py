import pandas as pd

# LangGraph system — builds and runs the multi-agent pipeline
from core.graph import build_graph

# ML pipeline — handles training, evaluation, and model improvement
from core.ml_pipeline import run_ml_pipeline, improve_model


def main():
    # Load a sample dataset from the internet (Iris dataset) for testing
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

    print("🚀 Running Multi-Agent System...\n")

    # Build and run the LangGraph multi-agent pipeline
    # Each agent (data analyst, feature engineer, etc.) runs in sequence
    app = build_graph()
    result = app.invoke({"df": df})

    # Print the output from each agent node
    for key, value in result.items():
        print(f"\n=== {key.upper()} ===")
        print(value)

    print("\n🚀 Running Real AutoML Pipeline...\n")

    # Run the actual ML training pipeline — trains LR, RF, SVM and picks the best
    ml_result = run_ml_pipeline(df)

    print("\n=== MODEL COMPARISON ===")

    # Print accuracy and F1 score for each trained model
    for model, metrics in ml_result["results"].items():
        print(f"\n{model}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    # Print the best model selected based on scoring
    print(f"\n🏆 BEST MODEL: {ml_result['best_model']}")
    print("📊 Graph saved as model_comparison.png")

    # 🔁 USER FEEDBACK LOOP — ask user if they are satisfied with the results
    while True:
        user_input = input("\nIs this accuracy sufficient? (yes/no): ").lower()

        if user_input == "yes":
            # User is happy — exit the loop
            print("✅ Final model accepted!")
            break

        elif user_input == "no":
            # User wants improvement — retrain with wider hyperparameter search
            improved_acc = improve_model(df)

            # Compare improved accuracy against the original best model
            best_acc = ml_result["results"][ml_result["best_model"]]["accuracy"]

            if improved_acc > best_acc:
                print("🔥 Model improved!")
            else:
                print("⚠️ No significant improvement.")

        else:
            print("Please type 'yes' or 'no'")


if __name__ == "__main__":
    main()

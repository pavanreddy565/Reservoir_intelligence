# Reservoir Intelligence

Reservoir Intelligence is a machine learning-powered system designed to **forecast water storage levels** and **assess supply-demand balance** for effective water resource management. It integrates historical water usage, climate projections, and external factors to provide actionable insights.

## Features

- **Dashboard Visualization**: Displays real-time reservoir levels and trends using Plotly for interactive plots.
- **Forecasting Models**: Predicts future **storage levels** and **outflow rates** using deep learning models like **LSTMs, GRUs, and Bidirectional GRUs**.
- **Supply-Demand Estimation**: Analyzes whether the water supply meets the demand based on historical and forecasted data.
- **Comparative Analysis**: Evaluates traditional models (ARIMA, SARIMA) against neural network-based approaches.
- **Decision Support**: Provides insights for policymakers to optimize **water management strategies** and **plan infrastructure**.

## Installation & Usage

### Prerequisites
Ensure you have Python installed on your system. If not, download it from [python.org](https://www.python.org/downloads/).

### Steps to Set Up

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reservoir-intelligence.git
   ```

2. Navigate to the project directory:
   ```bash
   cd reservoir-intelligence
   ```

3. Create and activate a virtual environment:
   ```bash
   pip install venv
   python -m venv myVenv
   .\myVenv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the dashboard:
   Open a new terminal, navigate to the project directory, and execute:
   ```bash
   python app.py
   ```

6. Access the dashboard by navigating to `http://127.0.0.1:5500/` in your web browser.

### Notes
- The system uses **Supabase** for data storage, ensuring seamless integration of real-time data.
- Interactive visualizations are powered by **Plotly**.

## Technologies Used

- **Python, Flask** – Backend & API handling
- **Plotly** – Data visualization
- **PyTorch** – Deep learning models
- **Pandas, NumPy** – Data processing
- **Supabase** – Data storage and management

## Contribution

Contributions are welcome! Feel free to submit issues or pull requests. Please ensure your contributions align with the project's goals and coding standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


"""Launch the FedPG-BR dashboard."""
from fedpg_br.dashboard.app import start_dashboard

if __name__ == "__main__":
    start_dashboard(host="127.0.0.1", port=5001)

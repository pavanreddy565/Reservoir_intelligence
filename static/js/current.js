// Define API endpoints
const apiEndpoints = {
    storage: 'http://127.0.0.1:5000/getWeekStorage',
    inflow: 'http://127.0.0.1:5000/getWeekInflow',
    outflow: 'http://127.0.0.1:5000/getWeekOutflow'
};

// Reservoir capacity data (used only for Storage normalization)
const reservoir_capacity = {
    'Chembarambakkam': 3645,
    'Cholavaram': 1081,
    'Poondi': 3231,
    'Redhills': 3300
};

// Colors for the plots
const colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'];

// Function to fetch data and update plots
async function fetchDataAndPlot(endpoint) {
    try {
        const response = await fetch(endpoint);
        const data = await response.json();

        // Extract the list of reservoirs from the data
        const reservoirs = Object.keys(data[0]).filter(key => key !== 'Date');
        const columns = Object.keys(data[0]);

            // Determine the endpoint type (Storage, Inflow, Outflow)
        const endpointType = endpoint === apiEndpoints.storage
                ? 'Storage'
                : endpoint === apiEndpoints.inflow
                    ? 'Inflow'
                    : 'Outflow';
        // Prepare data for Plotly bar chart
        const barTraces = reservoirs.map((reservoir, index) => {
            return {
                x: data.map(record => record.Date),
                y: endpoint === apiEndpoints.storage
                    ? data.map(record => (record[reservoir] / reservoir_capacity[reservoir]) * 100) // Normalize for Storage
                    : data.map(record => record[reservoir]), // No normalization for Inflow/Outflow
                name: reservoir,
                marker: { color: colors[index] },
                type: 'bar'
            };
        });

        // Prepare data for Plotly line plot
        const lineTraces = reservoirs.map((reservoir, index) => {
            return {
                x: data.map(record => record.Date),
                y: data.map(record => record[reservoir]), // No normalization for any data in line plot
                name: reservoir,
                mode: 'lines+markers',
                line: { color: colors[index], width: 2 }
            };
        });

        // Layout for the Bar Chart
        const barLayout = {
            barmode: 'group',
            title: {
                text: endpoint === apiEndpoints.storage
                    ? "Storage Levels of Different Reservoirs (Normalized)"
                    : endpoint === apiEndpoints.inflow
                        ? "Inflow Levels of Different Reservoirs (Cusecs)"
                        : "Outflow Levels of Different Reservoirs (Cusecs)",
                font: { size: 16 },
                x: 0.5,
                xanchor: 'center'
            },
            height: 300,
            width: 600,
            margin: { t: 60, b: 60, l: 70, },
            xaxis: { title: 'Date' },
            yaxis: { 
                title: endpoint === apiEndpoints.storage 
                    ? 'Storage %' 
                    : 'Cusecs' 
            }
        };

        // Layout for the Line Plot
        const lineLayout = {
            title: {
                text: endpoint === apiEndpoints.storage
                    ? "Storage Levels of Different Reservoirs (Original Values)"
                    : endpoint === apiEndpoints.inflow
                        ? "Inflow Levels of Different Reservoirs (Cusecs)"
                        : "Outflow Levels of Different Reservoirs (Cusecs)",
                font: { size: 16 },
                x: 0.5,
                xanchor: 'center'
            },
            height: 300,
            width: 600,
            margin: { t: 60, b: 60, l: 70, },
            xaxis: { title: 'Date' },
            yaxis: { 
                title: endpoint === apiEndpoints.storage 
                    ? 'Storage Value' 
                    : 'Cusecs' 
            }
        };

        // Update the plots
        Plotly.react('barChart', barTraces, barLayout);
        Plotly.react('linePlot', lineTraces, lineLayout);
        populateTable(data, columns, endpointType);
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}
// Function to populate the table
// Function to populate the table
function populateTable(data, columns, endpointType) {
    const table = document.getElementById('data-table');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');

    // Clear existing headers and rows
    thead.innerHTML = '';
    tbody.innerHTML = '';

    // Reverse the data array to display in reverse order (bottom-up)
    data = [...data].reverse();

    // Add headers
    const headerRow = document.createElement('tr');

    // First header cell for "Date"
    const dateHeader = document.createElement('th');
    dateHeader.textContent = 'Date';
    dateHeader.style.padding = '10px';
    dateHeader.style.background = '#f4f4f4';
    dateHeader.style.border = '1px solid #ddd';
    headerRow.appendChild(dateHeader);

    // Add reservoir headers with units
    columns.forEach(column => {
        if (column !== 'Date') {
            const th = document.createElement('th');
            const unit = endpointType === 'Storage' ? '(mcft)' : '(cusecs)';
            th.textContent = `${column} ${endpointType} ${unit}`;
            th.style.padding = '10px';
            th.style.background = '#f4f4f4';
            th.style.border = '1px solid #ddd';
            headerRow.appendChild(th);
        }
    });
    thead.appendChild(headerRow);

    // Add rows
    data.forEach(record => {
        const row = document.createElement('tr');

        // First cell for "Date"
        const dateCell = document.createElement('td');
        dateCell.textContent = record['Date'];
        dateCell.style.padding = '10px';
        dateCell.style.border = '1px solid #ddd';
        row.appendChild(dateCell);

        // Add reservoir data
        columns.forEach(column => {
            if (column !== 'Date') {
                const cell = document.createElement('td');
                cell.textContent = record[column];
                cell.style.padding = '10px';
                cell.style.border = '1px solid #ddd';
                row.appendChild(cell);
            }
        });

        tbody.appendChild(row);
    });
}


// Event listeners for buttons
document.getElementById('storageBtn').addEventListener('click', () => {
    setActiveButton('storageBtn');
    fetchDataAndPlot(apiEndpoints.storage);
});

document.getElementById('inflowBtn').addEventListener('click', () => {
    setActiveButton('inflowBtn');
    fetchDataAndPlot(apiEndpoints.inflow);
});

document.getElementById('outflowBtn').addEventListener('click', () => {
    setActiveButton('outflowBtn');
    fetchDataAndPlot(apiEndpoints.outflow);
});

// Function to set the active button
function setActiveButton(activeId) {
    document.querySelectorAll('.button-container button').forEach(button => {
        button.classList.remove('active');
    });
    document.getElementById(activeId).classList.add('active');
}

// Initial load: Fetch and plot storage data by default
fetchDataAndPlot(apiEndpoints.storage);
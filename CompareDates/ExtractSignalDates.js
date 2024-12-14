const fs = require('fs');

// Path to the JSON file
const filePath = './marked.json';

// Read and process the JSON file
fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }

    try {
        const jsonData = JSON.parse(data);
        const priceDataObj = jsonData.pricesInfo.priceDataObj;
        const result = [];

        // Iterate over the price data to find entries with the "signal" key
        for (const [date, details] of Object.entries(priceDataObj)) {
            if (details.signal) {
                result.push({ date, signal: details.signal });
            }
        }

        // Output the result
        console.log(result);
    } catch (parseErr) {
        console.error('Error parsing JSON:', parseErr);
    }
});

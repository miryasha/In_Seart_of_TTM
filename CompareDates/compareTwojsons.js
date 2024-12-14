// Function to read data from JavaScript files
const readDataFromFile = (filePath) => {
  try {
    // Use `require` to load the JS file directly
    return require(filePath);
  } catch (err) {
    console.error(`Error reading file ${filePath}:`, err);
    process.exit(1);
  }
};

// Function to compare the two datasets
const compareData = (trainedData, generatedData) => {
  let matchCount = 0;

  trainedData.forEach((trainedRecord) => {
    const match = generatedData.find(
      (generatedRecord) =>
        generatedRecord.date === trainedRecord.date &&
        generatedRecord.signal === trainedRecord.signal
    );
    if (match) matchCount++;
  });

  const totalEntries = trainedData.length;
  const percentageMatch = ((matchCount / totalEntries) * 100).toFixed(2);

  return { matchCount, totalEntries, percentageMatch };
};

// Main script execution
const main = () => {
  // Adjusting the file paths to include `.js` for requiring JavaScript files
  const trainedData = readDataFromFile('./JustOutPut.js'); // No need to parse as JSON
  const generatedData = readDataFromFile('./generated.js'); // No need to parse as JSON

  const { matchCount, totalEntries, percentageMatch } = compareData(
    trainedData,
    generatedData
  );

  console.log(`Matched Records: ${matchCount}`);
  console.log(`Total Records: ${totalEntries}`);
  console.log(`Percentage Match: ${percentageMatch}%`);
};

main();

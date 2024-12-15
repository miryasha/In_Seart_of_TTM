const TTMScalper = require('./TTMScalper');  // Assuming TTMScalper is in a separate file

function formatCandles(priceDataObj, priceDateArray) {
    return priceDateArray.map(date => ({
        open: parseFloat(priceDataObj[date]["1. open"]),
        high: parseFloat(priceDataObj[date]["2. high"]),
        low: parseFloat(priceDataObj[date]["3. low"]),
        close: parseFloat(priceDataObj[date]["4. close"]),
        date: date
    }));
}

function criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArray) {
    // Initialize TTM Scalper
    const ttmScalper = new TTMScalper();
    
    // Format candles data
    const candles = formatCandles(priceDataObj, priceDateArray);
    
    // Get TTM Scalper signals
    const signals = ttmScalper.analyze(candles);
    
    // Process and format results
    const results = {
        buySignals: [],
        sellSignals: [],
        summary: {
            totalBuySignals: 0,
            totalSellSignals: 0,
            lastSignal: null,
            currentTrend: null
        }
    };
    
    // Process signals
    signals.forEach(signal => {
        const signalData = {
            date: priceDateArray[signal.index],
            price: signal.price,
            trend: signal.trend === 1 ? 'Up' : 'Down'
        };
        
        if (signal.type === 'BUY') {
            results.buySignals.push(signalData);
            results.summary.totalBuySignals++;
        } else {
            results.sellSignals.push(signalData);
            results.summary.totalSellSignals++;
        }
        
        // Track last signal
        if (signal.index === signals[signals.length - 1]?.index) {
            results.summary.lastSignal = {
                type: signal.type,
                ...signalData
            };
            results.summary.currentTrend = signal.trend === 1 ? 'Up' : 'Down';
        }
    });
    
    // Add additional analysis
    results.analysis = {
        signalDistribution: {
            buyPercentage: (results.summary.totalBuySignals / signals.length * 100).toFixed(2),
            sellPercentage: (results.summary.totalSellSignals / signals.length * 100).toFixed(2)
        },
        recentSignals: signals.slice(-5).map(signal => ({
            type: signal.type,
            date: priceDateArray[signal.index],
            price: signal.price,
            trend: signal.trend === 1 ? 'Up' : 'Down'
        }))
    };
    
    return results;
}

// Update your route handler to use the new criteriaCheck function
router.post('/', async (req, res) => {
    try {
        const { ticker, marketName, standardDeviationFrom, timeFrame } = req.body;

        // Validate timeFrame
        const validTimeFrames = ['daily', 'weekly', 'monthly'];
        const normalizedTimeFrame = timeFrame.toLowerCase();
        if (!validTimeFrames.includes(normalizedTimeFrame)) {
            throw new Error(`Invalid timeFrame: ${timeFrame}. Must be one of: ${validTimeFrames.join(', ')}`);
        }

        const pricesInfo = await getPrice(ticker, marketName, normalizedTimeFrame);
        const indicators = await getIndicators(ticker, timeFrame);
        const backTestFrom = pricesInfo.priceDateArry.length - 1;

        const results = criteriaCheck(
            backTestFrom, 
            indicators, 
            pricesInfo.priceDataObj, 
            pricesInfo.priceDateArry
        );

        res.json({ results });
    } catch (error) {
        res.status(500).json({ 
            error: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

module.exports = router;
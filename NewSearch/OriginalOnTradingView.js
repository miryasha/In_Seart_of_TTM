// ////////////////////////////////////////////////////////////
// //  Copyright by HPotter v2.0 29/09/2014
// // TTM scalper indicator of John Carter’s Scalper Buys and Sells. The methodology 
// // is a close approximation of the one described in his book Mastering the Trade. 
// // The book is highly recommended. Note the squares are not real-time but will 
// // show up once the third bar has confirmed a reversal. 
// ////////////////////////////////////////////////////////////
// study(title="TTM scalper indicator", overlay = true)
// width = input(2, minval=1)
// triggerSell = iff(iff(close[1] < close,1,0) and (close[2] < close[1] or close[3] <close[1]),1,0)
// triggerBuy = iff(iff(close[1] > close,1,0) and (close[2] > close[1] or close[3] > close[1]),1,0)
// buySellSwitch = iff(triggerSell, 1, iff(triggerBuy, 0, nz(buySellSwitch[1])))
// SBS = iff(triggerSell and buySellSwitch[1] == false, high, iff(triggerBuy and buySellSwitch[1], low, nz(SBS[1])))
// clr_s = iff(triggerSell and buySellSwitch[1] == false, 1, iff(triggerBuy and buySellSwitch[1], 0, nz(clr_s[1])))
// clr = iff(clr_s == 0 , green , red)
// plot(SBS, color=clr, title="TTM", style = circles, linewidth = width)
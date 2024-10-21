import React, { useState } from "react";
import { useGlobalContext } from '../../Context/globalContext';

const MonthlyReport = () => {
  const { getMonthlyTransactions } = useGlobalContext();
  const [month, setMonth] = useState(9); // Example: September
  const [year, setYear] = useState(2024); // Example: 2024

  const {
    filteredIncomes,
    filteredExpenses,
    totalMonthlyIncome,
    totalMonthlyExpenses,
    totalTransactions,
    balance,
  } = getMonthlyTransactions(month, year);

  return (
    <div>
      <h2>Monthly Report for {month}/{year}</h2>
      <p>Total Transactions: {totalTransactions}</p>
      <p>Total Income: ${totalMonthlyIncome}</p>
      <p>Total Expenses: ${totalMonthlyExpenses}</p>
      <p>Balance: ${balance}</p>

      {/* Render the incomes and expenses here if needed */}
    </div>
  );
};

export default MonthlyReport;

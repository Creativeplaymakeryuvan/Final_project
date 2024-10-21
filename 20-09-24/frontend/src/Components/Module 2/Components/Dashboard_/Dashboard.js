import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useGlobalContext } from '../../Context/globalContext';
import SortBy from '../SortBy/SortBy';
import Cardinfo from '../Cardinfo/Cardinfo';
import History from '../History/History';
import BarChartComponent from '../BarChartComponent/BarChartComponent';
import PieChart from '../PieChartComponent/PieChartComponent';
import './dashboard.css';

const Dashboard = () => {
  const [name, setName] = useState('');
  const [totalSavings, setTotalSavings] = useState(0);
  const [totalSavingsbefore, setTotalSavingsbefore] = useState(0);
  const URL = 'http://localhost:3001/api/v1/';

  const getName = async () => {
    try {
      const userId = localStorage.getItem('userId');
      const response = await axios.get(`${URL}getName`, {
        headers: {
          'user-id': userId,
        },
      });
      setName(response.data.name);
    } catch (error) {
      console.error('Error fetching name:', error);
    }
  };

  useEffect(() => {
    getName();
  }, []);

  const { getMonthlyTransactions } = useGlobalContext();
  const { getCumulativeSavingsBeforeMonth } = useGlobalContext();

  const currentMonth = new Date().getMonth() + 1;
  const currentYear = new Date().getFullYear();

  const [sortBy, setSortBy] = useState(localStorage.getItem('sortBy') || 'custom');
  const [month, setMonth] = useState(currentMonth);
  const [year, setYear] = useState(currentYear);
  const [flag, setFlag] = useState(localStorage.getItem('sortBy') !== 'all');

  useEffect(() => {
    const cumulativeSavings = getCumulativeSavingsBeforeMonth(month, year);
    const cumulativeSavingsbefore = getCumulativeSavingsBeforeMonth(month - 1, year);

    if (cumulativeSavings !== totalSavings) {
        setTotalSavings(cumulativeSavings);
    }
    if (cumulativeSavingsbefore !== totalSavingsbefore) {
        setTotalSavingsbefore(cumulativeSavingsbefore);
    }
}, [sortBy, month, year, totalSavings, totalSavingsbefore, getCumulativeSavingsBeforeMonth]);

  

  const handleSortByChange = (selectedSort) => {
    setSortBy(selectedSort);

    if (selectedSort === 'all') {
      setMonth(month);
      setYear(year);
      setFlag(false);
    } else {
      setMonth(currentMonth);
      setYear(currentYear);
      setFlag(true);
    }
  };

  const handleMonthChange = (month) => {
    setMonth(month);
  };

  const handleYearChange = (year) => {
    setYear(year);
  };

  const handleSlideRight = () => {
    if (month === 1) {
      setMonth(12);
      setYear(year - 1);
    } else {
      setMonth(month - 1);
    }
  };

  const handleSlideLeft = () => {
    if (month === 12) {
      setMonth(1);
      setYear(year + 1);
    } else {
      setMonth(month + 1);
    }
  };

  const {
    filteredIncomes,
    filteredExpenses,
    totalMonthlyIncome,
    totalMonthlyExpenses,
    balance,
    filteredtransactionHistory
  } = getMonthlyTransactions(month, year);

  return (
    <div className="InnerLayout">
      <div className="head">
        <h2 className="name">Hi, {name}✌
          <p className="d-p">
            Here's what's happening with your money, let's manage your expenses.
          </p>
        </h2>
        <SortBy
          sortBy={sortBy}
          handleSortByChange={handleSortByChange}
          handleMonthChange={handleMonthChange}
          handleYearChange={handleYearChange}
          currentMonth={currentMonth}
          currentYear={currentYear}
          month={month}
          year={year}
        />
      </div>
      <div className="card-stats-con">
        <div className="chart-con">
          <div className="bar-chart-navigation">
            <div className='bar-chart-btn'>
              <button
                className="bar-chart-nav-btn"
                onClick={handleSlideRight}
                disabled={sortBy === 'all'}
              >←</button>
              <span className="month-display">
                {new Date(year, month - 1).toLocaleString('default', { month: 'long' })}
              </span>
              <button
                className="bar-chart-nav-btn"
                onClick={handleSlideLeft}
                disabled={sortBy === 'all'}
              >→</button>
            </div>
            <BarChartComponent
              incomeData={filteredIncomes}
              expenseData={filteredExpenses}
              flag={flag}
            />
          </div>
          <Cardinfo className="cardinfo-container"
            totalMonthlyIncome={totalMonthlyIncome}
            totalMonthlyExpenses={totalMonthlyExpenses}
            balance={balance}
            totalSavings={totalSavings} 
            totalSavingsbefore={totalSavingsbefore}
            flag={flag}
          />
        </div>
        <div className="history-con">
          <History className="history-container"
            filteredtransactionHistory={filteredtransactionHistory}
            flag={flag}
          />
          <PieChart className="pie-chart-container"
            incomeData={totalMonthlyIncome}
            expenseData={totalMonthlyExpenses}
            flag={flag}
          />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

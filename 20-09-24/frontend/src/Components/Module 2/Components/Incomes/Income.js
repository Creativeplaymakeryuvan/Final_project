import React, { useEffect } from 'react';
import { useGlobalContext } from '../../Context/globalContext';
import Form from '../Form/Form';
import IncomeItem from '../Incomeitem/incomeItem';
import './income.css';

function Income() {
  const { getIncomes, incomes, deleteIncome, totalIncome } = useGlobalContext();

  useEffect(() => {
    getIncomes();
  }, []);

  return (
    <div className='InnerLayout'>
      <h1>Incomes</h1>
      <h2 className='total-income'>Total Income: <span>₹{totalIncome()}</span></h2>
      <div className="Income-div">
        <div className='income-content'>
          <div className='form-container'>
            <Form />
          </div>
          <div className='incomes'>
            {incomes.length === 0 ? (
              <div className='no-data'>
                <span className='no-data-icon'>💰</span>
                <div className='no-transactions-message'>No Income data Available</div>
              </div>
            ) : (
              incomes.slice()
                .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
                .map((income) => {
                  const { _id, title, amount, date, type, category, description } = income;
                  return (
                    <IncomeItem
                      key={_id}
                      id={_id}
                      title={title}
                      description={description}
                      amount={amount}
                      date={date}
                      category={category}
                      indicatorColor="#42AD00"
                      type={type}
                      deleteItem={deleteIncome}
                    />
                  );
                })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Income;

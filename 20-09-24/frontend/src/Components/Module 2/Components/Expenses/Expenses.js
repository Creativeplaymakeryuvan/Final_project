import React, { useEffect } from 'react';
import { useGlobalContext } from '../../Context/globalContext';
import ExpenseItem from '../Incomeitem/incomeItem';
import './expenses.css';
import ExpenseForm from './ExpenseForm';

function Expenses() {
  const { getExpenses, expenses, deleteExpense, totalExpenses } = useGlobalContext();

  useEffect(() => {
    getExpenses();
  }, []);

  return (
    <div className='InnerLayout'>
      <h1>Expenses</h1>
      <h2 className='total-income'>Total Expense: <span>₹{totalExpenses()}</span></h2>
      <div className="Income-div">
        <div className='income-content'>
          <div className='form-container'>
            <ExpenseForm />
          </div>
          <div className='incomes'>
            {expenses.length === 0 ? (
              <div className='no-data'>
                <span className='no-data-icon'>💸</span>
                <div className='no-transactions-message'>No Expenses data Available</div>
              </div>
            ) : (
              expenses
                .slice()
                .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
                .map((expense) => {
                  const { _id, title, amount, date, type, category, description } = expense;
                  return (
                    <ExpenseItem
                      key={_id}
                      id={_id}
                      title={title}
                      description={description}
                      amount={amount}
                      date={date}
                      category={category}
                      indicatorColor="#42AD00"
                      deleteItem={deleteExpense}
                      type={type}
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

export default Expenses;

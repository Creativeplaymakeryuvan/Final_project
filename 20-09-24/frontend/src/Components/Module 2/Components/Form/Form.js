import React, { useState } from 'react';
import './form.css';
import { useGlobalContext } from '../../Context/globalContext';
import Button from '../Button/Button';

function Form() {
  const { addIncome, getIncomes, error, setError } = useGlobalContext();
  const userId = localStorage.getItem('userId');

  const currentDate = new Date().toISOString().split('T')[0]; 

  const [inputState, setInputState] = useState({
    title: '',
    amount: '',
    date: currentDate,
    category: '',
    subCategory: '',
    description: '',
    userId: userId,
  });

  const { title, amount, date, category, subCategory, description } = inputState;

  const categoriesData = {
    salary: [],
    freelancing: ['Project 1', 'Project 2', 'Consulting'],
    investments: ['Stocks', 'Bonds', 'Real Estate'],
    bank: ['Savings', 'Checking', 'Wire Transfer'],
    youtube: ['Sponsorship', 'Membership', 'Adsense'],
    other: ['Gift', 'Bonus', 'Other Income'],
  };

  const handleInput = (name) => (e) => {
    setInputState({ ...inputState, [name]: e.target.value });
  };

  const handleCategoryChange = (e) => {
    setInputState({ ...inputState, category: e.target.value, subCategory: '' });
  };

  const handleSubCategoryChange = (e) => {
    setInputState({ ...inputState, subCategory: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const finalCategory = subCategory ? `${category} - ${subCategory}` : category;

    const finalInputState = { ...inputState, category: finalCategory };

    addIncome(finalInputState);
    getIncomes();

    setInputState({
      title: '',
      amount: '',
      date: currentDate,
      category: '',
      subCategory: '',
      description: '',
      userId: userId,
    });
  };

  return (
    <form className="Form-div" onSubmit={handleSubmit}>
      {error && <p className="error">{error}</p>}

      <div className="input-control">
        <input
          type="text"
          value={title}
          name="title"
          placeholder="Income Title"
          onChange={handleInput('title')}
        />
      </div>

      <div className="input-control">
        <input
          type="text"
          value={amount}
          name="amount"
          placeholder="Income Amount"
          onChange={handleInput('amount')}
        />
      </div>

      <div className="input-control">
        <input
          type="date"
          value={date}
          onChange={(event) => setInputState({ ...inputState, date: event.target.value })}
        />
      </div>

      <div className="selects input-control">
        <select
          required
          value={category}
          name="category"
          id="category"
          onChange={handleCategoryChange}
        >
          <option value="" disabled>Select Option</option>
          {Object.keys(categoriesData).map((categoryOption, index) => (
            <option key={index} value={categoryOption}>
              {categoryOption.charAt(0).toUpperCase() + categoryOption.slice(1)}
            </option>
          ))}
        </select>

        {/* Subcategory select dropdown */}
        {category && categoriesData[category].length > 0 && (
          <div className="subcategories">
            <select
              value={subCategory}
              name="subCategory"
              id="subCategory"
              onChange={handleSubCategoryChange}
            >
              <option value="">Select Subcategory</option>
              {categoriesData[category].map((subOption, index) => (
                <option key={index} value={subOption}>
                  {subOption}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="input-control">
        <textarea
          name="description"
          value={description}
          placeholder="Add A Reference"
          id="description"
          cols="30"
          rows="4"
          onChange={handleInput('description')}
        ></textarea>
      </div>

      <div className="submit-btn-income">
        <Button
          name={'Add Income'}
          icon={<i className="fa-solid fa-plus"></i>}
          bPad={'.8rem 1.6rem'}
          bRad={'30px'}
          bg={'#F56692'}
          color={'#fff'}
        />
      </div>
    </form>
  );
}

export default Form;

import React, { useState } from 'react';
import './incomeItem.css';
import Button from '../Button/Button';
import { SlCalender } from "react-icons/sl";
import { FaCommentDots, FaTrash, FaRupeeSign, FaRegEye } from "react-icons/fa";
import { MdLocalGroceryStore } from "react-icons/md";
import { dateFormet } from '../../Utils/dateFormet';
import { BiCategory } from "react-icons/bi";
import { MdFastfood } from "react-icons/md";
import UpdateModal from '../Models/Modal';
import { useGlobalContext } from '../../Context/globalContext';

const IncomeItem = ({
  id,
  title,
  amount,
  date,
  category,
  description,
  deleteItem,
  indicaterColor,
  type
}) => {

  const { updateIncome, updateExpense } = useGlobalContext();
  const spilt_category = category.split(" - ")[1] ? category.split(" - ")[1] : category.split(" - ")[0];
  const userId = localStorage.getItem('userId');

  const [showModal, setShowModal] = useState(false);
  const [isEditable, setIsEditable] = useState(false);

  const categoryIcon = () => {
    switch (spilt_category) {
      case 'salary':
      case 'Bonus':
      case 'Commission':
      case 'Other Income':
        return <i className="fa-solid fa-money-bill"></i>;
      case 'freelancing':
      case 'Project 1':
      case 'Project 2':
      case 'Consulting':
        return <i className="fa-solid fa-earth-americas"></i>;
      case 'investments':
      case 'Stocks':
      case 'Bonds':
      case 'Real Estate':
        return <i className="fa-solid fa-arrow-trend-up"></i>;
      case 'bank':
      case 'Savings':
      case 'Checking':
      case 'Wire Transfer':
        return <i className="fa-brands fa-cc-visa"></i>;
      case 'youtube':
      case 'Sponsorship':
      case 'Membership':
      case 'Adsense':
        return <i className="fa-brands fa-youtube"></i>;
      case 'other':
      case 'Gift':
      case 'Bonus':
      case 'Other':
        return <i className="fa-solid fa-piggy-bank"></i>;
      default:
        return '';
    }
  };

  const expenseCatIcon = () => {
    switch (spilt_category) {
      case 'education':
      case 'Tuition':
      case 'Books':
      case 'Supplies':
        return <i className="fa-solid fa-book-open"></i>;
      case 'groceries':
      case 'Fruits':
      case 'Vegetables':
      case 'Dairy':
        return <i><MdLocalGroceryStore /></i>;
      case 'food':
      case 'Restaurants':
      case 'Snacks':
      case 'Beverages':
        return <i><MdFastfood /></i>
      case 'health':
      case 'Medicine':
      case 'Consultation':
      case 'Fitness':
        return <i className="fa-solid fa-briefcase-medical"></i>;
      case 'subscriptions':
      case 'Streaming':
      case 'Magazines':
      case 'Memberships':
        return <i className="fa-solid fa-tv"></i>;
      case 'clothing':
      case 'Shirts':
      case 'Pants':
      case 'Accessories':
        return <i className="fa-solid fa-shirt"></i>;
      case 'travelling':
      case 'Flights':
      case 'Hotels':
      case 'Transportation':
        return <i className="fa-solid fa-earth-americas"></i>;
      case 'other':
      case 'Miscellaneous':
      case 'Uncategorized':
        return <i className="fa-solid fa-circle-dot"></i>;
      default:
        return '';
    }
  };

  const handleView = () => {
    setIsEditable(false); 
    setShowModal(true);
  };

  const toggleEditable = () => {
    setIsEditable((prevEditable) => !prevEditable);
  };

  const handleClose = () => {
    setShowModal(false);
  };

  const handleSubmit = (updatedData) => {
    if ( type === 'income' ) {
        updateIncome(userId, id, updatedData);
    } else {
        updateExpense(userId, id, updatedData);
    }
    handleClose();
};

  return (
    <>
      <div className='incomeItem-div'>
        <div className="icon">
          {type === 'expense' ? expenseCatIcon() : categoryIcon()}
        </div>
        <div className="content">
          <h5>{title}</h5>
          <div className="inner-content">
            <div className="text">
              <p>{<FaRupeeSign />} {amount}</p>
              <p>{<SlCalender />} {dateFormet(date)}</p>
              <p>{<BiCategory />}{spilt_category}</p>
              <p className='item-description'>
                {<FaCommentDots />} {description.length > 5 ? `${description.substring(0, 5)}...` : description}
              </p>
            </div>
            <div className="btn-con">
              <Button
                className="view"
                icon={<FaRegEye />}
                bPad={'1rem'}
                bRad={'50%'}
                bg={'#222260'}
                color={'#fff'}
                iColor={'#fff'}
                hColor={'#42AD00'}
                onClick={handleView}
              />
              <Button
                icon={<FaTrash />}
                bPad={'1rem'}
                bRad={'50%'}
                bg={'#222260'}
                color={'#fff'}
                iColor={'#fff'}
                hColor={'#42AD00'}
                onClick={() => deleteItem(userId, id)}
              />
            </div>
          </div>
        </div>
      </div>

      <UpdateModal
        show={showModal}
        handleClose={handleClose}
        currentData={{ id, title, amount, date, category, description, type }}
        isEditable={isEditable}
        toggleEditable={toggleEditable}
        onSubmit={handleSubmit}
      />
    </>
  );
};

export default IncomeItem;

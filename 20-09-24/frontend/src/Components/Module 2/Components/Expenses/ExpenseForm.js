import React, { useState, useEffect } from 'react';
import { useGlobalContext } from '../../Context/globalContext';
import Button from '../Button/Button';
import { Modal, Button as BootstrapButton, Form as BootstrapForm } from 'react-bootstrap';
import './expenseForm.css';
import UpdateModal from '../Models/Modal';

function ExpenseForm() {
  const { addExpense, getExpenses, error, setError, processBillImage, processAndExtractReceiptData, uploadImage } = useGlobalContext();
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

  const [showImageModal, setShowImageModal] = useState(false);
  const [showUpdateModal, setShowUpdateModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [billDataState, setBillDataState] = useState(null);
  const [selectedOption, setSelectedOption] = useState(null);

  const { title, amount, date, category, subCategory, description } = inputState;

  const categoriesData = {
    education: ['Tuition', 'Books', 'Supplies'],
    groceries: ['Fruits', 'Vegetables', 'Dairy'],
    food: ['Restaurants', 'Snacks', 'Beverages'],
    health: ['Medicine', 'Consultation', 'Fitness'],
    subscriptions: ['Streaming', 'Magazines', 'Memberships'],
    clothing: ['Shirts', 'Pants', 'Accessories'],
    travelling: ['Flights', 'Hotels', 'Transportation'],
    other: ['Miscellaneous', 'Uncategorized'],
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
    addExpense(finalInputState);
    getExpenses();

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

  const handleShow = () => setShowImageModal(true);
  const handleClose = () => setShowImageModal(false);
  

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const fileType = file.type;
      if (fileType.startsWith('image/') || fileType === 'application/pdf') {
        setSelectedFile(file);
      } else {
        alert('Please upload a valid image file (jpg, jpeg, png) or PDF file.');
        setSelectedFile(null);
      }
    } else {
      alert('No file selected.');
      setSelectedFile(null);
    }
  };

  const handleImageSubmit = async (e) => {
    e.preventDefault();
    if (selectedFile) {
      const billData = await processBillImage(selectedFile);
      if (billData) {
        console.log('Processed Bill Data:', billData);
        setBillDataState(billData)
        setShowUpdateModal(true);
      } else {
        alert('Error processing bill image');
      }
    }
  };

  const handleUpdateSubmit = (updatedBillData) => {
    console.log('Updated Bill Data:', updatedBillData);
      const tax = updatedBillData.tax
      const descriptionString = updatedBillData.description
      .map(({ item, quantity, price }) => `${item}-Q:${quantity}-Rs:${price}`)
      .join('\n') + `\nTax-Rs:${tax}`;


    const expenseData = {
      title: updatedBillData.title,
      amount: updatedBillData.amount,
      date: updatedBillData.date,
      category: updatedBillData.category,
      subCategory: updatedBillData.subCategory,
      description: descriptionString,
      userId: userId,
    };


    addExpense(expenseData);
        setShowUpdateModal(false);
      };

  return (
    <div>
      <form className="Form-div" onSubmit={handleSubmit}>
        {error && <p className="error">{error}</p>}

        <div className="input-control">
          <input
            type="text"
            value={title}
            name="title"
            placeholder="Expense Title"
            onChange={handleInput('title')}
          />
        </div>

        <div className="input-control">
          <input
            type="text"
            value={amount}
            name="amount"
            placeholder="Expense Amount"
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

        <div className="submit-btn-expense">
          <Button
            name={'Add Expense'}
            icon={<i className="fa-solid fa-plus"></i>}
            bPad={'.8rem 1.6rem'}
            bRad={'30px'}
            bg={'#F56692'}
            color={'#fff'}
          />

          <BootstrapButton variant="primary" onClick={handleShow} className="add-image-btn">
            Add Image
          </BootstrapButton>
        </div>
      </form>

      <Modal show={showImageModal} onHide={handleClose} className="custom-modal">
        <Modal.Header closeButton className="modal-header-custom">
          <Modal.Title className="modal-title-custom">Upload Your Receipt Image</Modal.Title>
        </Modal.Header>
        <Modal.Body className="modal-body-custom">
          <p className="modal-prompt">Please upload your Receipt image below for reference:</p>
          {/* Separate form for Image Upload */}
          <BootstrapForm onSubmit={handleImageSubmit}>
            <BootstrapForm.Group>
              <BootstrapForm.Label>Select Image</BootstrapForm.Label>
              <BootstrapForm.Control
                type="file"
                accept=".jpg,.jpeg,.png,.pdf"
                onChange={handleFileChange}
                required
              />
            </BootstrapForm.Group>

            {/* <br />
            <BootstrapForm.Group>
              <BootstrapForm.Label className="custom-label">Select an Option</BootstrapForm.Label>
              <BootstrapForm.Control
                as="select"
                value={selectedOption}
                onChange={handleSelectChange}
                className="custom-select"
                required
              >
                <option value="">Choose...</option>
                <option value="receipt">Receipt</option>  
                <option value="invoice">Invoice</option>
              </BootstrapForm.Control>
            </BootstrapForm.Group> */}

            <div className="upload-btn-wrapper">
              <BootstrapButton variant="primary" type="submit" className="upload-btn">
                Upload Image
              </BootstrapButton>
            </div>
          </BootstrapForm>
        </Modal.Body>
      </Modal>

      {billDataState && (
        <UpdateModal
          show={showUpdateModal}
          handleClose={() => {
            setShowUpdateModal(false)
            setBillDataState(null);
          }}
          currentData={billDataState}
          onSubmit={handleUpdateSubmit}
        />
      )}
    </div>
  );
}

export default ExpenseForm;

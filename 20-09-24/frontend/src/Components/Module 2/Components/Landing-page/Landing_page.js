import React from 'react'
import './landing_page.css'
import Orb from '../Orbs/Orb'
import Navigation from '../Navigation/Navigation'
import Dashboard from '../Dashboard_/Dashboard'
import Transaction from '../Transactions/Transaction'
import Income from '../Incomes/Income'
import Expenses from '../Expenses/Expenses'
import { useState } from 'react';
import { useGlobalContext } from '../../Context/globalContext'
import Budget from '../../../Module 3/components/Budget/Budjet'


function Landing_page() {

  const [active, setActive] = React.useState(1)

  const global = useGlobalContext()
  console.log(global)  

  const displayData = () => {
    switch (active) {
      case 1:
        return <Dashboard />
      // case 2:
      //   return <Budget />
      case 2:
        return <Transaction />
      case 3: 
        return <Income />
      case 4:
        return <Expenses />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className='App-div'>
      <Orb />
      <div className='MainLayout'>
        <Navigation active={active} setActive={setActive} />
        <main>
          {displayData()}
        </main>
      </div>
      <div className='InnerLayout'>
      </div>
    </div>
  )
}

export default Landing_page

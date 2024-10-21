import React, { useEffect, useState } from 'react'
import './login.css'
import video from '../../Imgs/Rain.mp4'
import logo from '../../Imgs/logo4.png'
import { Link } from 'react-router-dom'
import { FaUserShield } from "react-icons/fa";
import { AiOutlineSwapRight } from "react-icons/ai";
import { BsFillShieldLockFill } from "react-icons/bs";
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const Login = ()=>{
    const [users,setUsers] = useState([])
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const navigate=useNavigate()

    useEffect(() => {
        fetchUsers();
    }, [])

    const fetchUsers = () => {
        const response = axios.get('http://localhost:3001/register')
        .then((res) => {
            console.log(res.data)
        })
    }

    const handleLogin = async (event) => {
        event.preventDefault();
        try{
            const response = await axios.post('http://localhost:3001/login', { username, password })
            localStorage.setItem('userId',response.data._id)
            alert('Login succesfull')
            setUsername('')
            setPassword('')
            fetchUsers();
            navigate('/landing_page')
            // window.location.reload()
            // localStorage.setItem('token', token)
        }
        catch(e){
            alert(e.response.data.error)
        }
    }
    return(
        <div className='login-page flex'>
           <div className='container flex'>

                <div className='videoDiv'>
                    <video src={video} autoPlay muted loop></video>
                    <div className='textDiv'>
                        <h2 className='title'>SmartSavings</h2>
                        <p>Beware of little expenses</p>
                    </div>
                    <div className='footerDiv flex'>
                        <span className='text'>Don't have an account?</span>
                        <Link to={'/register'}>
                            <button className='btn'>Sign Up</button>
                        </Link>
                    </div>
                </div> 

                <div className="formDiv flex">
                    <div className="headerDiv">
                        <img src={logo} alt="Logo Img" />
                        <h3>Welcome Back!</h3>
                    </div>
                    <form action="" onSubmit={handleLogin} className='form grid'>

                        <span className='showMessage'>Login Status will go here</span>
                        <div className="inputDiv">
                            <label htmlFor="username">Username</label>
                            <div className="input flex">
                            <FaUserShield className='icon'/>
                            <input 
                                 type="text" 
                                 id='userName' 
                                 placeholder='Enter Username'
                                 onChange={(e) => setUsername(e.target.value)}
                                 required
                            />
                            </div>
                        </div>


                        <div className="inputDiv">
                            <label htmlFor="password">Password</label>
                            <div className="input flex">
                            <BsFillShieldLockFill className='icon'/>
                            <input 
                                type="password" 
                                id='password' 
                                placeholder='Enter Password' 
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                            </div>
                        </div>

                        <button type='submit' className='btn flex'>
                            <span>Login</span>
                            <AiOutlineSwapRight className='icon'/>
                        </button>

                    </form>
                </div>
           </div>
        </div>
    )
}

export default Login
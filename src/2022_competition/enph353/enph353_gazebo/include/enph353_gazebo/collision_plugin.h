#ifndef COLLISION_PLUGIN_HH
#define COLLISION_PLUGIN_HH

#include <gazebo/common/Plugin.hh>
#include <gazebo/transport/transport.hh>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

namespace gazebo
{

class CollisionPlugin : public WorldPlugin
{
public:
  // Methods

  /// Constructor.
  CollisionPlugin();

  /// Documentation inherited.
  /// Called when a Plugin is first created, and after the World has been
  /// loaded. This function should not be blocking.
  /// \param[in] _world Pointer the World
  /// \param[in] _sdf Pointer the the SDF element of the plugin.
  void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf);

  /// \brief Callback function when collision occurs in the world.
  /// \param[in] _contacts List of all collisions from last simulation iteration
  private: void OnCollisionMsg(ConstContactsPtr &_contacts);

  /////////////////////////////////////////////////////////////////////////
  // Fields
  /// \brief World
  private: physics::WorldPtr world;

  /// \brief Collision detection node subscriber
  private: transport::SubscriberPtr collisionSub;

  /// \brief gazebo node pointer
  private: transport::NodePtr collisionNode;

  /// \brief ROS node handle.
  private: std::unique_ptr<ros::NodeHandle> rosNode;

  /// \brief Publisher for the collision.
  private: ros::Publisher contactPub;

  /// \brief ROS Contact Bool Msg.
  private: std_msgs::Bool contactMsg;

  /// \brief Last time that the contact was published
  private: gazebo::common::Time lastPubTime;

  /// \brief Period between publishes
  private: double pubPeriod;

  /// \brief Topic to publish to 
  private: std::string pubTopic;

  /// \brief Collision object to look for
  private: std::string collisionObject1;
};

}  // namespace gazebo

#endif

